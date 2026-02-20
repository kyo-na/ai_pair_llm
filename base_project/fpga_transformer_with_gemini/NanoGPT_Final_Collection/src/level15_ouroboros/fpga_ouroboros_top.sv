module fpga_ouroboros_top (
    input wire clk, input wire btn1, output wire uart_tx, output wire [5:0] led
);
    wire rst_n = btn1;
    reg bank_sel; reg [31:0] gen_count;
    wire eng_start, eng_done, we_eng;
    wire [9:0] addr_w; wire [1:0] w_val;
    wire [4:0] addr_x, addr_y; wire [31:0] data_x, data_y;
    
    // Engine & ROM
    wire [15:0] d_a; bitnet_rom w_rom (.addr(addr_w), .data_a(d_a), .weight_b(w_val));
    vector_engine eng (.clk(clk), .rst_n(rst_n), .start(eng_start), .addr_w(addr_w), .weight_val(w_val), .addr_x(addr_x), .data_x(data_x), .we_y(we_eng), .addr_y(addr_y), .data_y(data_y), .done(eng_done));

    // RAM (Double Buffer)
    logic [31:0] ram [0:63]; logic [31:0] r_data;
    initial begin for(int i=0;i<64;i++) ram[i]=0; ram[0]=100; ram[1]=50; ram[5]=-50; end
    always_ff @(posedge clk) begin if (we_eng) ram[{~bank_sel, addr_y}]<=data_y; r_data<=ram[{bank_sel, addr_x}]; end
    assign data_x = r_data;

    // Control & UART
    typedef enum logic [2:0] {START, WAIT, PREP, SEND, FLIP} st_t; st_t state;
    reg [4:0] tx_idx; reg tx_req; reg [7:0] tx_d; wire tx_busy;
    uart_tx tx (.clk(clk), .rst_n(rst_n), .start(tx_req), .data_in(tx_d), .tx_serial(uart_tx), .busy(tx_busy));
    assign eng_start = (state==START);
    assign led = ~gen_count[5:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin state<=START; bank_sel<=0; gen_count<=0; end
        else case (state)
            START: state<=WAIT;
            WAIT: if (eng_done) begin tx_idx<=0; state<=PREP; end
            PREP: begin tx_d<=ram[{~bank_sel, tx_idx}][7:0]; tx_req<=1; state<=SEND; end
            SEND: begin tx_req<=0; if (!tx_busy && !tx_req) begin if (tx_idx==31) state<=FLIP; else begin tx_idx<=tx_idx+1; state<=PREP; end end end
            FLIP: begin bank_sel<=~bank_sel; gen_count<=gen_count+1; state<=START; end
        endcase
    end
endmodule
