from os import system, remove
from sys import argv

system(f"sed -nr 's/^[^#]/mpiexec -n {argv[1]} /p' test.sh > t")
system("./test.sh > x.txt")
x = open("x.txt", "r").readlines()[14::15]
o = open("original.txt", "r").readlines()
remove("x.txt")
remove("t")

diff = [i for i in range(len(x)) if x[i] != o[i]]
[print(f"Fallo en el test {i + 1}.") for i in diff]
print(f"\n{len(x)} tests realizados.")
