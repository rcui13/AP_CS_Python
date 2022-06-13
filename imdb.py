def phraseinfile(file, phrase):
    count = 0
    openfile = open(file)
    print(openfile.readline().replace(",", " "))
    for line in openfile:
        if line.lower().find(phrase) != -1:
            print(line.replace(",", " "))
            count += 1
    print(str(count) + " matches.")

term = input("Search Term:")
phraseinfile("imdb.txt", term)
# out = open("imdb.txt", "a")
# out.write("1,196376,9.1,The Shawshank Redemp+++++++++++++++++++++++++++++                      +++++++++++++++++++++++++++++++++
