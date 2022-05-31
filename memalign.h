#include <malloc.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#define memalign(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free _aligned_free
#else
#define aligned_free free
#endif
