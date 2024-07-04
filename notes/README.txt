+ Train 1
Allenato con SGD per 10 epoche, momentum = 0.9 e learning rate 0.01. Apparentemente sembra
scendere, ma in maniera molto instabile.

+ Train 2
Allenato con Adam per 20 epoche
opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
sembra andare addirittura peggio di prima.
Nonostante stia trainando con il batch norm la discesa della loss è estremamente instabile,
bisogna trovare una soluzione.

+ Train 3
opt = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.999)
epochs = 30
Plottando la loss media per epoca sembra scendere alla fine. Direi che si debba solo trainare per più epoche,
servono molte ore su questa macchina.

+ Train 4
Nonostante la loss sia scesa a 0.4, quindi manca ancora un po', l'ouput è qualcosa di sensato. Per ora ci stiamo
limitando a Postdam, non abbiamo ancora considerato le immagini low-quality dell'ESA.
Devo provare a trainare per almeno 150 epoche, ci vuole tipo un giorno in locale. Serve accedere ai server di Baraldi al più presto.

+Train 5
epochs = 150
opt = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.999)
Il modello, un checkpoint e i valori della loss sono stati salvati in locale.
La loss è scesa fino a 0.079!
Alcuni esempi di testing si possono trovare nella cartella train5.
Il modello è stato validato: l'mIoU è pari a 51,3% (in linea con i valori riportati nel paper di Bahr (ovviamente per l'Unet classica), 
lui ottiene un 53% ma traina per molto più tempo e con anche i dati di DeepGlobe). 
Se invece si tiene conto del non bilanciamento delle label, la mIoU è pari al 78,3%
