


def checkFile():
    """retrieve the content of temp.txt for export module """
    checkfile=os.path.isfile('temp.txt')
    if(checkfile==True):
        fr=open("temp.txt","r")
        content=fr.read()
        fr.close()
    else:
        content="No Content Available"
    return content

def removeFile():
    """Removes the temp.txt and tempgest directory if any stop button is pressed oor application is closed"""
    try:
        os.remove("temp.txt")
    except:
        pass
    try:
        shutil.rmtree("TempGest")
    except:
        pass