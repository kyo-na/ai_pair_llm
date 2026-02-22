= NanoGPT Final Collection =

Congratulations on completing the course!

[Folder Structure]
src/
  tangnano9k.cst        : Common Pin Constraints
  uart_tx.sv            : Common UART Module
  
  level13_gpt/          : Level 13 (Standard Nano-GPT)
    - fpga_gpt_top.sv   : Top Module (Set this as Top in Gowin)
    
  level15_ouroboros/    : Level 15 (BitNet Ouroboros)
    - fpga_ouroboros_top.sv : Top Module (Set this as Top in Gowin)

python/
  recv_matrix.py        : Use for Level 13
  recv_stream.py        : Use for Level 15

[How to Build]
1. Create a new Gowin Project.
2. Add 'src/tangnano9k.cst' and 'src/uart_tx.sv'.
3. Choose either Level 13 or Level 15, and add ALL files from that folder.
4. Set the corresponding '_top.sv' as Top Module.
5. Synthesize, Place & Route, and Program.

[How to Run]
1. Connect Tang Nano 9K.
2. Run the corresponding Python script (check COM port).
3. Release Reset button on FPGA.

Good luck with your future FPGA journey!
