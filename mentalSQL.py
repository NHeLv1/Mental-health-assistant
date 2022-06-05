import sqlite3
import time
import matplotlib.pyplot as plt
import os
import numpy as np


def setNewDatabaseOrSheet(file_name: str):
    conn = sqlite3.connect(f'{file_name}.db')
    c = conn.cursor()
    localtime = time.localtime(time.time())
    y = localtime.tm_year
    m = localtime.tm_mon
    d = localtime.tm_mday
    try:
        c.execute(f'''CREATE TABLE Y{y}M{m}D{d}
               (ID INTEGER     NOT NULL,
               tired           INTEGER,
               angry           INTEGER,
               sad             INTEGER,
               neutral         INTEGER,
               happy           INTEGER,
               surprised       INTEGER,
               disgust         INTEGER,
               scared          INTEGER,
               PRIMARY KEY(ID AUTOINCREMENT));''')

        ins_sql = f'insert into Y{y}M{m}D{d} DEFAULT VALUES'
        for i in range(24 * 60):
            c.execute(ins_sql)
        conn.commit()

    except:
        pass
    conn.close()


def getMonthData(file_name: str, year, month):
    conn = sqlite3.connect(f'{file_name}.db')
    c = conn.cursor()
    c.execute("select name from sqlite_master where type='table' order by name")
    l = map(lambda x: x[0].split('D'), c.fetchall())
    s = set()
    for i in l:
        if len(i) > 1 and i[0] == f"Y{year}M{month}":
            s.add(int(i[1]))

    MonthData = []
    for n in range(1, 32):
        if n in s:
            c.execute(f'SELECT * FROM Y{year}M{month}D{n}')
            a, result = 0, [0, 0, 0, 0, 0, 0, 0, 0]
            for d in c.fetchall():
                if d[1]:
                    a += 1
                    r = []
                    for x, y in zip(result, d[1:]):
                        r.append(x + y)
                    result = r

            MonthData.append((n, tuple(map(lambda x: x // a, result))))

    conn.close()

    return MonthData


def updateData(file_name: str, tired, mental_data):
    conn = sqlite3.connect(f'{file_name}.db')
    c = conn.cursor()
    localtime = time.localtime(time.time())
    y = localtime.tm_year
    mon = localtime.tm_mon
    d = localtime.tm_mday
    h = localtime.tm_hour
    minute = localtime.tm_min

    up_sql = f'''UPDATE Y{y}M{mon}D{d} set tired = {tired},
                                    angry={mental_data[0]},
                                    sad={mental_data[1]},
                                    neutral={mental_data[2]},
                                    happy={mental_data[3]},
                                    surprised={mental_data[4]},
                                    disgust={mental_data[5]},
                                    scared={mental_data[6]} where ID={h * 60 + minute + 1}'''

    c.execute(up_sql)
    conn.commit()
    conn.close()


def ID2Date(id):
    return f"{(id - 1) // 60}:{(id - 1) % 60:02}"


def drawTheMentalData(title, data):
    ls = os.listdir("guiData/bars")
    for i in ls:
        c_path = os.path.join("guiData/bars", i)
        os.remove(c_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

    color = ('#2F4F4F', '#008B8B', '#8A2BE2', '#00FFFF', '#00FA9A', "#4682B4", '#1E90FF', "#B0C4DE")
    mental_kind = ['疲倦', '愤怒', '伤心', '中立', '开心', '惊奇', '厌恶', '恐惧']

    labels=mental_kind

    t_enable = []
    for i in data:
        if i[1] is not None:
            t_enable.append(i)
    group_t_enable = []
    for i in range(len(t_enable) - 1):
        if i == 0:
            group_t_enable.append([t_enable[i]])
        if t_enable[i + 1][0] - t_enable[i][0] >= 5:
            group_t_enable.append([t_enable[i + 1]])
        else:
            top = len(group_t_enable) - 1
            group_t_enable[top].append(t_enable[i + 1])

    for group in group_t_enable:
        avg_list = [0, 0, 0, 0, 0, 0, 0, 0]
        for sample in group:
            for i in range(1, len(sample)):
                avg_list[i-1] += sample[i]
        avg_list = list(map(lambda x:x/len(group),avg_list))
        plt.title(title+f'{str(ID2Date(group[0][0]))}~{str(ID2Date(group[-1][0]))}', fontproperties="SimHei")
        wedgeprops = {'width': 0.3, 'edgecolor': 'white', 'linewidth': 1}
        #plt.bar(labels,avg_list,color=color)
        plt.pie(avg_list, wedgeprops=wedgeprops, startangle=0, colors=color)
        plt.legend(labels, loc='upper left')
        plt.savefig(f'guiData/bars/{str(group[0][0])}.jpg', dpi=200, bbox_inches='tight')
        plt.close()

    #for n, x in enumerate(zip(color, mental_kind)):
        #plt.plot(Tdata[0], Tdata[n + 1], color=x[0], label=x[1])

    #plt.legend()  # 显示图例

    #plt.ylabel('心情值')
    #plt.xlabel('时间')
    #plt.savefig('guiData/chart.jpg', dpi=200, bbox_inches='tight')
    #plt.close()
'''

while True:
    setNewDatabaseOrSheet('text')
    updateData('text', 12 , (11,1111,1111,111,11,11,11,11,11,11))

    conn = sqlite3.connect('text.db')
    sql = 'SELECT * FROM Y2022M4D6'
    sur = conn.cursor()
    sur.execute(sql)

    for i in sur.fetchall():
        if i[1]: print(ID2Date(i[0]),i)

    print()
    print(getMonthData('text',2022,4))

    print()
    time.sleep(60)

'''
