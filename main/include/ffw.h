#ifndef FFW_H
#define FFW_H

#include "layer.h"
#include <deque>

class FFW {
  public:
    FFW()=default;
    void add(Layer layer);
  private:
    std::deque<Layer> layers;
};

#endif
