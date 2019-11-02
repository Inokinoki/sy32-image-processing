
def calculate_cover_rate(rect1:tuple, rect2:tuple) -> float:
    """
    rect i, j, h, l
    """
    p_start = (max((rect1[0], rect2[0])), max((rect1[1], rect2[1])))
    p_end = (min((rect1[0]+rect1[2], rect2[0]+rect2[2])), min((rect1[1]+rect1[3], rect2[1]+rect2[3])))

    cover_square = abs(p_end[1]-p_start[1]) * abs(p_end[0]-p_start[0])

    total_square = rect1[2] * rect1[3] + rect2[2] * rect2[3] - cover_square

    if total_square != 0:
        return cover_square/total_square
    else:
        return 1.0

def cluster(labels:list) -> dict:
    d = dict()
    for label in labels:
        if label[0] not in d.keys():
            l = list()
            l.append(label)
            d.update({label[0]: l})
        else:
            d[label[0]].append(label)
    
    return d


def get_labels(path:str, types:tuple) -> list:
    f_label = open(path)

    datas = []

    line:str = f_label.readline()
    while line:
        line = line.replace("\n", "")
        data = line.split(" ")
        for i, t in zip(range(len(data)), types):
            data[i] = float(data[i])
            data[i] = t(data[i])
        datas.append(data)

        line = f_label.readline()

    f_label.close()
    return datas

if __name__ == "__main__":
    labels:list = get_labels("/home/inoki/.sy32/TP3/results_train_500.txt", (int, float, float, float, float, float))

    grouped_labels:dict = cluster(labels)

    print([len(grouped_labels[k]) for k in grouped_labels.keys()])

    #new_labels:list = list()

    for index in grouped_labels.keys():
        i_base = 0
        while i_base < len(grouped_labels[index]):
            i_comp = i_base + 1
            while i_comp < len(grouped_labels[index]):
                cover_rate = calculate_cover_rate(grouped_labels[index][i_base][1:-1], grouped_labels[index][i_comp][1:-1])
                if cover_rate > 0.5:
                    if grouped_labels[index][i_comp][-1] > grouped_labels[index][i_base][-1]:
                        grouped_labels[index].remove(grouped_labels[index][i_comp])
                    elif grouped_labels[index][i_comp][-1] < grouped_labels[index][i_base][-1]:
                        grouped_labels[index].remove(grouped_labels[index][i_base])
                i_comp += 1

            i_base += 1
        """
        for i_base in range(len(grouped_labels[index])):
            
            for i_comp in range(len(i_base, grouped_labels[index])):

                if i_comp != i_base:
                    cover_rate = calculate_cover_rate(grouped_labels[index][i_base][1:-1], grouped_labels[index][i_comp][1:-1])
                    if cover_rate >= 0.5:
                        if grouped_labels[index][i_comp][-1] > grouped_labels[index][i_base][-1]:
                            
                        else:
                            new_labels.append(grouped_labels[index][i_base])
                    else:
                        # Add all to the new list
                        
            new_labels.append(grouped_labels[index][i_comp])
            """
    print([len(grouped_labels[k]) for k in grouped_labels.keys()])





