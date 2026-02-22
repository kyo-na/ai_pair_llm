module ternary_mac (
    input logic signed [31:0] acc_in, input logic signed [15:0] val_a, input logic [1:0] w_val, output logic signed [31:0] acc_out
);
    always_comb case(w_val) 0:acc_out=acc_in; 1:acc_out=acc_in+val_a; 2:acc_out=acc_in-val_a; default:acc_out=acc_in; endcase
endmodule
