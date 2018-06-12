def logger(directory, filename, string):
    f = open(directory + '/' + filename, 'a')
    f.write(string + '\n')
    f.close()