#include <vector>
#include <string>
#include <iterator> 

class PNG {
public:
	unsigned w, h;
	std::vector<unsigned char> data;
	PNG() {}
	PNG(unsigned w, unsigned h) { Create(w, h); }
	PNG(std::string file) { Load(file); }
	unsigned Load(std::string file);
	unsigned Save(std::string file);
	void Create(unsigned w, unsigned h);
	void Free();
};
