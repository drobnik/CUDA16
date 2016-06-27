#include "png.h"
#include "lodepng.h"

unsigned PNG::Load(std::string file) {
	Free();
	return lodepng::decode(data, w, h, file.c_str());
}

unsigned PNG::Save(std::string file) {
		return lodepng::encode(file.c_str(), data, w, h);
}

void PNG::Create(unsigned w, unsigned h) {
		Free();
		this->w = w;
		this->h = h;
		data.reserve(w * h * 4);
}

void PNG::Free() {
		w = 0;
		h = 0;
		data.clear();
}