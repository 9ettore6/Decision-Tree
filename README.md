# Decision-Tree
Implementazione dell'algoritmo decision-tree-learning su tre dataset
 
 Intelligenza Artificiale 
 Celozzi Ettore Maria 5963792

-Dal sito aima(https://github.com/aimacode/aima-python/blob/master/learning.py) sono 
 state riprese le funzioni per l'apprendimento, la costruzione dell'albero e per la 
 costruzione del dataset alle quali sono state apportate delle modifiche per adattarle
 al problema;

-La versione di Python utilizzata è la 2.7;

-Le librerie utilizzate sono: 1 "matplotlib" per creare i grafici;
			      2 "print_function" per usare la stampa di python 3.x;
			      3 "random" per mischiare gli esempi ad ogni iterazione;

-Per l'esecuzione del programma basta eseguire il file "Testing.py", infatti all'interno
 di questo sono contenuti attributi e target di ogni dataset. Una volta eseguito
 il programma richiederà di scegliere tra i vari dataset, di selezionare il valore di
 m_range(nonostante sia stato scelto un valore per i risulatati presentati nella relazione,
 ho voluto lasciare questa libertà in quanto un valore elevato può impiegare molto tempo) 
 e la scelta del tipo di pruning da utilizzare.
 Una volta eseguito, il programma mostrerà le curve di apprendimento e infine l'albero 
 che si ottiene con 1 esecuzione dell'algoritmo di learning senza pre-pruning.
