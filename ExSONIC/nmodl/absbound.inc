FUNCTION absBound(x, y) {
    if (x < 0) {
        if (x < -y) {
            x = -y
        }
    } else {
        if (x > y) {
            x = y
        }
    }
    absBound = x
}