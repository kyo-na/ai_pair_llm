module matrix_engine (
    input logic clk, rst_n, start, input logic [7:0] base_addr_a, base_addr_b, base_addr_c,
    output logic [7:0] addr_a, addr_b, output logic we_c, output logic [7:0] addr_c, output logic [31:0] wdata_c, output logic done,
    input logic [31:0] rdata_a, rdata_b
);
    localparam SIZE = 4; logic [2:0] i, j, k; logic [31:0] acc;
    typedef enum logic [2:0] { IDLE, READ, WAIT, CALC, WRITE, NEXT } state_t; state_t state;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin state <= IDLE; done <= 0; we_c <= 0; end
        else case (state)
            IDLE: begin done<=0; we_c<=0; if (start) begin i<=0; j<=0; k<=0; acc<=0; state<=READ; end end
            READ: begin addr_a<=base_addr_a+(i<<2)+k; addr_b<=base_addr_b+(k<<2)+j; state<=WAIT; end
            WAIT: state<=CALC;
            CALC: begin acc<=acc+(rdata_a*rdata_b); if (k==SIZE-1) state<=WRITE; else begin k<=k+1; state<=READ; end end
            WRITE: begin we_c<=1; addr_c<=base_addr_c+(i<<2)+j; wdata_c<=acc; state<=NEXT; end
            NEXT: begin we_c<=0; acc<=0; k<=0; if (j==SIZE-1) begin j<=0; if (i==SIZE-1) begin done<=1; state<=IDLE; end else begin i<=i+1; state<=READ; end end else begin j<=j+1; state<=READ; end end
        endcase
    end
endmodule
