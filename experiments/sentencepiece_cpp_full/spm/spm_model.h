#pragma once
#include <string>
#include <vector>
enum PieceType{NORMAL=1,UNKNOWN=2,CONTROL=3,BYTE=4};
struct SentencePiece{std::string piece;float score;PieceType type;};
struct SPMModel{std::vector<SentencePiece> pieces;int unk_id=-1;void load(const char*);};
