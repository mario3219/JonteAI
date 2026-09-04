#include "layer.h"
#include "ffw.h"

void FFW::add(Layer layer) {
  layers.push_front(layer);
}
