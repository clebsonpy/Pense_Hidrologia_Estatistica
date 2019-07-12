from random import randint

def apresentador(n_apresentador_passado, n_dia_seguidos_apresentado):
    list_nome = ['Clebson', 'Rafael', 'Wallisson']
    i = 0
    while i != n_dia_seguidos_apresentado+1:
        n = randint(1,3)
        if n != n_apresentador_passado:
            return list_nome[n-1], n
        
        i += 1
    return list_nome[n-1], n

n_apresentador_passado = int(input("Apresentador Passado: "))
n_dia_seguidos_apresentado = int(input("Dias seguidos apresentado: "))
apresentador, n = apresentador(n_apresentador_passado, n_dia_seguidos_apresentado)
print('Apresentador: %s - %s' % (n, apresentador))
