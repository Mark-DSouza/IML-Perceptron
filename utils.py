def load_class_data(filename):
    class_data = list()

    with open(filename, 'r') as file:
        lines  = file.readlines()
        for line in lines:
            row = [ float(x) for x in line.split() ]
            class_data.append(row)

    return class_data