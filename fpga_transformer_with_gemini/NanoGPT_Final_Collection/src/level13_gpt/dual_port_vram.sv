module dual_port_vram (
    input logic clk, we_a, input logic [7:0] addr_a, input logic [31:0] wdata_a, output logic [31:0] rdata_a,
    input logic we_b, input logic [7:0] addr_b, input logic [31:0] wdata_b, output logic [31:0] rdata_b
);
    logic [31:0] mem [0:255];
    initial begin
        for (int i=0; i<256; i++) mem[i]=0;
        mem[0]=1; mem[5]=1; mem[10]=1; mem[15]=1; // Q=Identity
        for (int i=16; i<32; i++) mem[i]=2; // K=2
        for (int i=32; i<48; i++) mem[i]=3; // V=3
    end
    always_ff @(posedge clk) begin if (we_a) mem[addr_a]<=wdata_a; rdata_a<=mem[addr_a]; end
    always_ff @(posedge clk) begin if (we_b) mem[addr_b]<=wdata_b; rdata_b<=mem[addr_b]; end
endmodule
