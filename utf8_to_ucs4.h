#include "stdint.h"

// adapted from fontconfig, licensed under MIT
inline int FcUtf8ToUcs4 (const char *src_orig, int32_t *dst, int len) {
    const char *src = src_orig;
    char s;
    int extra;
    int32_t result;
    
    if (len == 0) {
        return 0;
    }
    
    s = *src++;
    len--;
    
    if (!(s & 0x80)) {
        result = s;
        extra = 0;
    } else if (!(s & 0x40)) {
        return -1;
    } else if (!(s & 0x20)) {
        result = s & 0x1f;
        extra = 1;
    } else if (!(s & 0x10)) {
        result = s & 0xf;
        extra = 2;
    } else if (!(s & 0x08)) {
        result = s & 0x07;
        extra = 3;
    } else if (!(s & 0x04)) {
        result = s & 0x03;
        extra = 4;
    } else if ( ! (s & 0x02)) {
        result = s & 0x01;
        extra = 5;
    } else {
        return -1;
    }
    if (extra > len) {
        return -1;
    }
    
    while (extra--) {
        result <<= 6;
        s = *src++;
    
        if ((s & 0xc0) != 0x80) {
            return -1;
        }
    
        result |= s & 0x3f;
    }
    *dst = result;
    return src - src_orig;
}

inline int utf8_to_ucs4(const char *src, int32_t *dst, int len) {
    int clen, len32 = 0;
    
    while (len > 0) {
        clen = FcUtf8ToUcs4(src, dst, len);
        if (clen <= 0) { /* malformed UTF8 string */
            return 0;
        }
    	src += clen;
    	len -= clen;
    	dst++;
    	len32++;
    }
    return len32;
}
