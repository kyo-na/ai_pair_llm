module softmax_engine (
    input logic clk, rst_n, start, input logic [7:0] base_addr, input logic [3:0] count,
    output logic [7:0] addr, input logic [31:0] rdata, output logic we, output logic [31:0] wdata, output logic done
);
    typedef enum logic [1:0] { IDLE, READ, WRITE, NEXT } state_t; state_t state;
    logic [3:0] i; logic [31:0] converted;
    softmax_rom my_rom (.x_in(rdata), .y_out(converted));
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin state<=IDLE; done<=0; we<=0; end
        else case (state)
            IDLE: begin done<=0; we<=0; if (start) begin i<=0; state<=READ; end end
            READ: begin we<=0; addr<=base_addr+i; state<=WRITE; end
            WRITE: begin we<=1; addr<=base_addr+i; wdata<=converted; state<=NEXT; end
            NEXT: begin we<=0; if (i==count-1) begin done<=1; state<=IDLE; end else begin i<=i+1; state<=READ; end end
        endcase
    end
endmodule
