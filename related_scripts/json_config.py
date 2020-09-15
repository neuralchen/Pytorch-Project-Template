import  json


def readConfig(path):
    with open(path,'r') as cf:
        nodelocaltionstr = cf.read()
        nodelocaltioninf = json.loads(nodelocaltionstr)
        if isinstance(nodelocaltioninf,str):
            nodelocaltioninf = json.loads(nodelocaltioninf)
    return nodelocaltioninf

def writeConfig(path, info):
    with open(path, 'w') as cf:
        configjson  = json.dumps(info, indent=4)
        cf.writelines(configjson)

if __name__ == "__main__":
    # wocao = {"name":"hehe","gender":"male"}
    # write_config("./file.json",wocao)
    wocao = readConfig("./model_config.json")
    for item in wocao.items():
        print("%s : %s"%(item[0],item[1]))
    pass