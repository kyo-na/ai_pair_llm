module vector_engine (
    input logic clk, rst_n, start,
    output logic [9:0] addr_w, input logic [1:0] weight_val,
    output logic [4:0] addr_x, input logic [31:0] data_x,
    output logic we_y, output logic [4:0] addr_y, output logic [31:0] data_y, output logic done
);
    logic [4:0] i, k; logic [31:0] acc, next_acc;
    ternary_mac mac (.acc_in(acc), .val_a(data_x[15:0]), .w_val(weight_val), .acc_out(next_acc));
    typedef enum logic [1:0] {IDLE, CALC, WRITE, NEXT} st_t; st_t state;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin state<=IDLE; done<=0; we_y<=0; end
        else case (state)
            IDLE: if (start) begin i<=0; k<=0; acc<=0; state<=CALC; end
            CALC: begin addr_w<={i,k}; addr_x<=k; if (k==31) state<=WRITE; else begin acc<=next_acc; k<=k+1; end end
            WRITE: begin we_y<=1; addr_y<=i; data_y<=next_acc; state<=NEXT; end
            NEXT: begin we_y<=0; acc<=0; k<=0; if (i==31) begin done<=1; state<=IDLE; end else begin i<=i+1; state<=CALC; end end
        endcase
    end
endmodule
