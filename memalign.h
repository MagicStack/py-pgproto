#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include <malloc.h>
#define memalign(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free _aligned_free
#else
#if __APPLE__
#include <stdlib.h>
#define memalign(alignment, size) valloc(size)
#else
#include <malloc.h>
#endif
#define aligned_free free
#endif
