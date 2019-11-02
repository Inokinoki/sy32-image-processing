import os

if __name__ == "__main__":
    files = os.listdir("validate/1")

    names = [False for i in range(0, 500)]

    for f in files:
        if f != "output":
            names[int(f[5:-4])-1] = True
            #print(f)

    for name_index in range(len(names)):
        if not names[name_index]:
            print(name_index+1)