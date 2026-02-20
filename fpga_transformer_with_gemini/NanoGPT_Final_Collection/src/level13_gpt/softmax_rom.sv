module softmax_rom (input logic [31:0] x_in, output logic [31:0] y_out);
    always_comb begin
        if (x_in[31]) y_out=0;
        else case (x_in[3:0])
            0:y_out=1; 1:y_out=2; 2:y_out=4; 3:y_out=8; 4:y_out=16; 5:y_out=32; 6:y_out=64; 7:y_out=128; default:y_out=255;
        endcase
    end
endmodule
