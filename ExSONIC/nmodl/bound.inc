FUNCTION absBound(x, xabsmax) {
    if (x < 0) {
        if (x < -xabsmax) {
            x = -xabsmax
        }
    } else {
        if (x > xabsmax) {
            x = xabsmax
        }
    }
    absBound = x
}