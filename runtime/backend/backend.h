
#pragma once
struct Backend {
    virtual void reset() = 0;
    virtual int step(int token) = 0;   // decode one token
    virtual bool finished() const = 0; // eos reached
    virtual ~Backend() {}
};
