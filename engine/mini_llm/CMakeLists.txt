cl /std:c++17 /O2 /utf-8 /EHsc ^
main.cpp ^
runtime\scheduler\scheduler.cpp ^
runtime\resource\resource_snapshot.cpp ^
runtime\threads\worker_pool.cpp ^
engine\mini_llm\src\model\mini_AI.cpp ^
engine\mini_llm\src\blocks\transformer_block4d.cpp ^
engine\mini_llm\src\layers\attention4d.cpp ^
engine\mini_llm\src\layers\layernorm4d.cpp ^
engine\mini_llm\src\layers\linear4d.cpp ^
engine\mini_llm\src\layers\embedding4d.cpp ^
engine\mini_llm\src\optimizer\adam.cpp ^
engine\mini_llm\src\world\world_model.cpp ^
engine\mini_llm\src\loss\mse4d.cpp ^
engine\mini_llm\src\loss\cross_entropy4d.cpp ^
engine\mini_llm\tokenizer\tokenizer.cpp ^
engine\mini_llm\tokenizer\vocab.cpp ^
engine\mini_llm\tokenizer\utf8.cpp ^
/I . ^
/I engine\mini_llm\include ^
/I engine\mini_llm\tokenizer ^
/Fe:ai_pair_llm.exe