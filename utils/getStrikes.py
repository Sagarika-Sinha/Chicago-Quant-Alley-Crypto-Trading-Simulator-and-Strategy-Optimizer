def getATMStrikes(price, strike_gap=1000):
    atm = round(price / strike_gap) * strike_gap
    return atm