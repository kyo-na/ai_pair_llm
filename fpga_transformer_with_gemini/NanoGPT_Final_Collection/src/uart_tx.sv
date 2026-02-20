module uart_tx #(
    parameter CLK_FREQ = 27000000,
    parameter BAUD_RATE = 9600
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,
    input  wire [7:0] data_in,
    output reg        tx_serial,
    output reg        busy
);
    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    typedef enum logic [1:0] { IDLE, START, DATA, STOP } state_t;
    state_t state;
    logic [15:0] clk_count;
    logic [2:0]  bit_index;
    logic [7:0]  tx_data;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; tx_serial <= 1; busy <= 0;
        end else begin
            case (state)
                IDLE: begin
                    tx_serial <= 1;
                    if (start) begin
                        state <= START; tx_data <= data_in; busy <= 1; clk_count <= 0;
                    end else busy <= 0;
                end
                START: begin
                    tx_serial <= 0;
                    if (clk_count < CLKS_PER_BIT - 1) clk_count <= clk_count + 1;
                    else begin clk_count <= 0; state <= DATA; bit_index <= 0; end
                end
                DATA: begin
                    tx_serial <= tx_data[bit_index];
                    if (clk_count < CLKS_PER_BIT - 1) clk_count <= clk_count + 1;
                    else begin
                        clk_count <= 0;
                        if (bit_index < 7) bit_index <= bit_index + 1;
                        else state <= STOP;
                    end
                end
                STOP: begin
                    tx_serial <= 1;
                    if (clk_count < CLKS_PER_BIT - 1) clk_count <= clk_count + 1;
                    else begin state <= IDLE; busy <= 0; end
                end
            endcase
        end
    end
endmodule
