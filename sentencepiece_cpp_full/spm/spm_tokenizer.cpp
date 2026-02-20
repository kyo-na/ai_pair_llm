#include "spm_tokenizer.h"
#include <algorithm>
std::vector<int> SPMTokenizer::encode(const std::string& text) const{int n=text.size();std::vector<float> dp(n+1,-1e9);std::vector<int> back(n+1,-1);dp[0]=0;for(int i=0;i<n;i++){if(dp[i]<-1e8)continue;for(int id=0;id<(int)model.pieces.size();id++){auto& p=model.pieces[id];if(text.compare(i,p.piece.size(),p.piece)==0){int j=i+p.piece.size();float s=dp[i]+p.score;if(s>dp[j]){dp[j]=s;back[j]=id;}}}}std::vector<int> ids;for(int i=n;i>0;){int id=back[i];if(id<0){ids.push_back(model.unk_id);break;}ids.push_back(id);i-=model.pieces[id].piece.size();}std::reverse(ids.begin(),ids.end());return ids;}
std::string SPMTokenizer::decode(const std::vector<int>& ids) const{std::string s;for(int id:ids)s+=model.pieces[id].piece;return s;}
