module bitnet_rom (
    input logic [9:0] addr, output logic [15:0] data_a, output logic [1:0] weight_b
);
    assign data_a = {12'd0, addr[3:0]};
    always_comb case(addr[1:0]) 0:weight_b=0; 1:weight_b=1; 2:weight_b=2; default:weight_b=1; endcase
endmodule
