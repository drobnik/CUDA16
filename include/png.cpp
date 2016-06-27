#include "p.h"
typedef struct LodePNGColorMode {
	LodePNGColorType colortype;
	unsigned bitdepth;
	unsigned char* palette;
	size_t palettesize;
	unsigned key_defined;
	unsigned key_r;
	unsigned key_g;
	unsigned key_b;
} LodePNGColorMode;

typedef enum LodePNGColorType {
	LCT_GREY = 0,
	LCT_RGB = 2,
	LCT_PALETTE = 3,
	LCT_GREY_ALPHA = 4,
	LCT_RGBA = 6
} LodePNGColorType;

typedef struct LodePNGState {
#ifdef LODEPNG_COMPILE_DECODER
	LodePNGDecoderSettings decoder;
#endif
#ifdef LODEPNG_COMPILE_ENCODER
	LodePNGEncoderSettings encoder;
#endif
	LodePNGColorMode info_raw;
	LodePNGInfo info_png;
	unsigned error;
#ifdef LODEPNG_COMPILE_CPP

	virtual ~LodePNGState() {}
#endif
} LodePNGState;

typedef struct LodePNGInfo {
	unsigned compression_method;
	unsigned filter_method;
	unsigned interlace_method;
	LodePNGColorMode color;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	unsigned background_defined;
	unsigned background_r;
	unsigned background_g;
	unsigned background_b;
	size_t text_num;
	char** text_keys;
	char** text_strings;
	size_t itext_num;
	char** itext_keys;
	char** itext_langtags;
	char** itext_transkeys;
	char** itext_strings;
	unsigned time_defined;
	LodePNGTime time;
	unsigned phys_defined;
	unsigned phys_x;
	unsigned phys_y;
	unsigned phys_unit;
	unsigned char* unknown_chunks_data[3];
	size_t unknown_chunks_size[3];
#endif
} LodePNGInfo;

unsigned PNG::Load(std::string file)
	{
		Free();
		return decode(data, w, h, file.c_str());
	}
unsigned PNG::Save(std::string file)
	{
		return encode(file.c_str(), data, w, h);
	}
	void PNG::Create(unsigned w, unsigned h)
	{
		Free();
		this->w = w;
		this->h = h;
		data.reserve(w * h * 4);
	}
	void PNG::Free()
	{
		w = 0;
		h = 0;
		data.clear();
	}

	unsigned encode(std::vector<unsigned char>& out,
		const unsigned char* in, unsigned w, unsigned h,//
		State& state)
	{
		unsigned char* buffer;
		size_t buffersize;
		unsigned error = lodepng_encode(&buffer, &buffersize, in, w, h, &state);
		if (buffer)
		{
			out.insert(out.end(), &buffer[0], &buffer[buffersize]);
			lodepng_free(buffer);
		}
		return error;
	}

	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, const std::string& filename,
		LodePNGColorType colortype, unsigned bitdepth)
	{
		std::vector<unsigned char> buffer;
		unsigned error = load_file(buffer, filename);
		if (error) return error;
		return decode(out, w, h, buffer, colortype, bitdepth);
	}
	unsigned lodepng_encode(unsigned char** out, size_t* outsize,
	const unsigned char* image, unsigned w, unsigned h,
	LodePNGState* state)
{
	LodePNGInfo info;
	ucvector outv;
	unsigned char* data = 0;
	size_t datasize = 0;


	*out = 0;
	*outsize = 0;
	state->error = 0;

	lodepng_info_init(&info);
	lodepng_info_copy(&info, &state->info_png);

	if ((info.color.colortype == LCT_PALETTE || state->encoder.force_palette)
		&& (info.color.palettesize == 0 || info.color.palettesize > 256))
	{
		state->error = 68;
		return state->error;
	}

	if (state->encoder.auto_convert)
	{
		state->error = lodepng_auto_choose_color(&info.color, image, w, h, &state->info_raw);
	}
	if (state->error) return state->error;

	if (state->encoder.zlibsettings.btype > 2)
	{
		CERROR_RETURN_ERROR(state->error, 61);
	}
	if (state->info_png.interlace_method > 1)
	{
		CERROR_RETURN_ERROR(state->error, 71);
	}

	state->error = checkColorValidity(info.color.colortype, info.color.bitdepth);
	if (state->error) return state->error;
	state->error = checkColorValidity(state->info_raw.colortype, state->info_raw.bitdepth);
	if (state->error) return state->error;

	if (!lodepng_color_mode_equal(&state->info_raw, &info.color))
	{
		unsigned char* converted;
		size_t size = (w * h * lodepng_get_bpp(&info.color) + 7) / 8;

		converted = (unsigned char*)lodepng_malloc(size);
		if (!converted && size) state->error = 83;
		if (!state->error)
		{
			state->error = lodepng_convert(converted, image, &info.color, &state->info_raw, w, h);
		}
		if (!state->error) preProcessScanlines(&data, &datasize, converted, w, h, &info, &state->encoder);
		lodepng_free(converted);
	}
	else preProcessScanlines(&data, &datasize, image, w, h, &info, &state->encoder);

	ucvector_init(&outv);
	while (!state->error)
	{
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
		size_t i;
#endif

		writeSignature(&outv);

		addChunk_IHDR(&outv, w, h, info.color.colortype, info.color.bitdepth, info.interlace_method);
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

		if (info.unknown_chunks_data[0])
		{
			state->error = addUnknownChunks(&outv, info.unknown_chunks_data[0], info.unknown_chunks_size[0]);
			if (state->error) break;
		}
#endif

		if (info.color.colortype == LCT_PALETTE)
		{
			addChunk_PLTE(&outv, &info.color);
		}
		if (state->encoder.force_palette && (info.color.colortype == LCT_RGB || info.color.colortype == LCT_RGBA))
		{
			addChunk_PLTE(&outv, &info.color);
		}

		if (info.color.colortype == LCT_PALETTE && getPaletteTranslucency(info.color.palette, info.color.palettesize) != 0)
		{
			addChunk_tRNS(&outv, &info.color);
		}
		if ((info.color.colortype == LCT_GREY || info.color.colortype == LCT_RGB) && info.color.key_defined)
		{
			addChunk_tRNS(&outv, &info.color);
		}
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

		if (info.background_defined) addChunk_bKGD(&outv, &info);

		if (info.phys_defined) addChunk_pHYs(&outv, &info);


		if (info.unknown_chunks_data[1])
		{
			state->error = addUnknownChunks(&outv, info.unknown_chunks_data[1], info.unknown_chunks_size[1]);
			if (state->error) break;
		}
#endif

		state->error = addChunk_IDAT(&outv, data, datasize, &state->encoder.zlibsettings);
		if (state->error) break;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

		if (info.time_defined) addChunk_tIME(&outv, &info.time);

		for (i = 0; i != info.text_num; ++i)
		{
			if (strlen(info.text_keys[i]) > 79)
			{
				state->error = 66;
				break;
			}
			if (strlen(info.text_keys[i]) < 1)
			{
				state->error = 67;
				break;
			}
			if (state->encoder.text_compression)
			{
				addChunk_zTXt(&outv, info.text_keys[i], info.text_strings[i], &state->encoder.zlibsettings);
			}
			else
			{
				addChunk_tEXt(&outv, info.text_keys[i], info.text_strings[i]);
			}
		}

		if (state->encoder.add_id)
		{
			unsigned alread_added_id_text = 0;
			for (i = 0; i != info.text_num; ++i)
			{
				if (!strcmp(info.text_keys[i], "LodePNG"))
				{
					alread_added_id_text = 1;
					break;
				}
			}
			if (alread_added_id_text == 0)
			{
				addChunk_tEXt(&outv, "LodePNG", LODEPNG_VERSION_STRING);
			}
		}

		for (i = 0; i != info.itext_num; ++i)
		{
			if (strlen(info.itext_keys[i]) > 79)
			{
				state->error = 66;
				break;
			}
			if (strlen(info.itext_keys[i]) < 1)
			{
				state->error = 67;
				break;
			}
			addChunk_iTXt(&outv, state->encoder.text_compression,
				info.itext_keys[i], info.itext_langtags[i], info.itext_transkeys[i], info.itext_strings[i],
				&state->encoder.zlibsettings);
		}


		if (info.unknown_chunks_data[2])
		{
			state->error = addUnknownChunks(&outv, info.unknown_chunks_data[2], info.unknown_chunks_size[2]);
			if (state->error) break;
		}
#endif
		addChunk_IEND(&outv);

		break;
	}

	lodepng_info_cleanup(&info);
	lodepng_free(data);

	*out = outv.data;
	*outsize = outv.size;

	return state->error;
}

void lodepng_info_cleanup(LodePNGInfo* info) {
	lodepng_color_mode_cleanup(&info->color);
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	LodePNGText_cleanup(info);
	LodePNGIText_cleanup(info);

	LodePNGUnknownChunks_cleanup(info);
#endif
}
unsigned load_file(std::vector<unsigned char>& buffer, const std::string& filename) {
	std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
	if (!file) return 78;


	std::streamsize size = 0;
	if (file.seekg(0, std::ios::end).good()) size = file.tellg();
	if (file.seekg(0, std::ios::beg).good()) size -= file.tellg();


	buffer.resize(size_t(size));
	if (size > 0) file.read((char*)(&buffer[0]), size);

	return 0;
}


unsigned save_file(const std::vector<unsigned char>& buffer, const std::string& filename) {
	std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
	if (!file) return 79;
	file.write(buffer.empty() ? 0 : (char*)&buffer[0], std::streamsize(buffer.size()));
	return 0;
}