module fpga_gpt_top (
    input  wire       clk,
    input  wire       btn1,
    output wire       uart_tx,
    output reg  [5:0] led
);
    wire rst_n = btn1;
    wire att_done, eng_start, eng_done, we_c;
    wire [7:0] base_a, base_b, base_c, addr_a, addr_b, addr_c;
    wire [31:0] rdata_a, rdata_b, wdata_c;
    wire sm_start, sm_done, sm_we;
    wire [7:0] sm_base, sm_addr;
    wire [31:0] sm_wdata;
    
    // UART Readout
    typedef enum logic [1:0] { WAIT_CALC, READ_MEM, SEND_UART, NEXT_BYTE, DONE_ALL } ro_state_t;
    ro_state_t ro_state;
    reg [4:0] send_counter;
    reg tx_start; reg [7:0] tx_byte; wire tx_busy; wire [31:0] read_data;

    attention_sequencer my_seq (.*, .start(rst_n), .done(att_done), .eng_addr_a_base(base_a), .eng_addr_b_base(base_b), .eng_addr_c_base(base_c), .sm_base_addr(sm_base));
    matrix_engine my_engine (.*, .start(eng_start), .done(eng_done), .base_addr_a(base_a), .base_addr_b(base_b), .base_addr_c(base_c));
    softmax_engine my_softmax (.*, .start(sm_start), .done(sm_done), .base_addr(sm_base), .count(4'd16), .addr(sm_addr), .rdata(rdata_a), .we(sm_we), .wdata(sm_wdata));

    logic use_softmax; assign use_softmax = (sm_start || !sm_done) && !eng_start;
    wire [7:0] mux_addr_a = use_softmax ? sm_addr : (we_c ? addr_c : addr_a);
    wire mux_we_a = use_softmax ? sm_we : we_c;
    wire [31:0] mux_wdata_a = use_softmax ? sm_wdata : wdata_c;

    dual_port_vram my_ram (.clk(clk), .we_a(mux_we_a), .addr_a(mux_addr_a), .wdata_a(mux_wdata_a), .rdata_a(rdata_a),
        .we_b(1'b0), .addr_b(ro_state == WAIT_CALC ? addr_b : (64 + send_counter)), .wdata_b(0), .rdata_b(rdata_b));
    assign read_data = rdata_b;

    uart_tx my_tx (.clk(clk), .rst_n(rst_n), .start(tx_start), .data_in(tx_byte), .tx_serial(uart_tx), .busy(tx_busy));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin ro_state <= WAIT_CALC; send_counter <= 0; tx_start <= 0; end
        else case (ro_state)
            WAIT_CALC: if (att_done) begin send_counter <= 0; ro_state <= READ_MEM; end
            READ_MEM: ro_state <= SEND_UART;
            SEND_UART: begin tx_byte <= read_data[7:0]; tx_start <= 1; ro_state <= NEXT_BYTE; end
            NEXT_BYTE: begin tx_start <= 0; if (!tx_busy && !tx_start) begin
                if (send_counter == 15) ro_state <= DONE_ALL; else begin send_counter <= send_counter + 1; ro_state <= READ_MEM; end
            end end
        endcase
    end
    always_comb begin
        if (ro_state == DONE_ALL) led = 6'b000000; else if (att_done) led = 6'b001100;
        else if (sm_start) led = 6'b101010; else led = 6'b010101;
    end
endmodule
