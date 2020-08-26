import math

def ChangeTimeFormat(time):
    day = 24*60*60
    hour = 60*60
    min = 60
    if time <60:
        return  "%d sec"%math.ceil(time)
    elif  time > day:
        days = divmod(time,day)
        return "%d days %s"%(int(days[0]),ChangeTimeFormat(days[1]))
    elif time > hour:
        hours = divmod(time,hour)
        return '%d hours %s'%(int(hours[0]),ChangeTimeFormat(hours[1]))
    else:
        mins = divmod(time,min)
        return "%d mins %d sec"%(int(mins[0]),math.ceil(mins[1]))

if __name__ == '__main__':
    time = 15611561561
    time = ChangeTimeFormat(time)
    print(time)