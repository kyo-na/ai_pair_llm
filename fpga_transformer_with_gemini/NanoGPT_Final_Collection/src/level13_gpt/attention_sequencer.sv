module attention_sequencer (
    input logic clk, rst_n, start, input logic eng_done, sm_done,
    output logic done, eng_start, sm_start,
    output logic [7:0] eng_addr_a_base, eng_addr_b_base, eng_addr_c_base, sm_base_addr
);
    typedef enum logic [2:0] { IDLE, M1_S, M1_W, SM_S, SM_W, M2_S, M2_W, FIN } state_t;
    state_t state;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin state <= IDLE; eng_start <= 0; sm_start <= 0; done <= 0; end
        else case (state)
            IDLE: begin done <= 0; if (start) state <= M1_S; end
            M1_S: begin eng_addr_a_base<=0; eng_addr_b_base<=16; eng_addr_c_base<=48; eng_start<=1; state<=M1_W; end
            M1_W: begin eng_start<=0; if (eng_done) state<=SM_S; end
            SM_S: begin sm_base_addr<=48; sm_start<=1; state<=SM_W; end
            SM_W: begin sm_start<=0; if (sm_done) state<=M2_S; end
            M2_S: begin eng_addr_a_base<=48; eng_addr_b_base<=32; eng_addr_c_base<=64; eng_start<=1; state<=M2_W; end
            M2_W: begin eng_start<=0; if (eng_done) state<=FIN; end
            FIN: begin done<=1; state<=IDLE; end
        endcase
    end
endmodule
