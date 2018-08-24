PROCEDURE update_stimon(){
    : Update the value of the boolean parameter stimon during ON and OFF periods throughout the
    : numerical integration of ASTIM simulations, according to stimulus parameters

    if(t < tstim){
        if(fmod(t, 1 / PRF) <= (DC / PRF) && stimon == 0){        : 0 -> 1 at pulse onset
            stimon = 1
        }else if(fmod(t, 1 / PRF) > (DC / PRF) && stimon == 1){   : 1 -> 0 at pulse offset
            stimon = 0
        }
    }else if(stimon == 1){                                        : 1 -> 0 at stimulus offset
        stimon = 0
    }
}