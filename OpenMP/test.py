from os import system, remove

system("./test.sh > x.txt")
x = open("x.txt", "r").readlines()[2::3]
o = open("original.txt", "r").readlines()
remove("x.txt")

diff = [i for i in range(len(x)) if x[i] != o[i]]
[print(f"Fallo en el test {i + 1}.") for i in diff]
print(f"\n{len(x) + 1} tests realizados.")
