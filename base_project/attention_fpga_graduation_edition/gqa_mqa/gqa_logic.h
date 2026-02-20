
#pragma once

inline int map_kv_head(int head, int num_heads, int num_kv_heads) {
    if (num_kv_heads == 1) return 0; // MQA
    int group = num_heads / num_kv_heads;
    return head / group;
}
