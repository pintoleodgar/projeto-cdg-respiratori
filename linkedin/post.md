# Post LinkedIn: projeto-cdg-respiratori

Treinei um modelo para detetar doença pulmonar em sons de auscultação. Deteta COPD sem falhar. E é por isso que lhe chamo protótipo, não solução.

O projeto usa o ICBHI 2017: 851 gravações, 104 pacientes, três classes. O primeiro modelo parecia ótimo. Depois fui ver porquê, e dois atalhos explicavam quase tudo:

- O equipamento. Saudáveis e URTI só foram gravados com um estetoscópio. O modelo aprendia a reconhecer o aparelho, não o pulmão.
- A duração. As gravações de COPD duravam 58 segundos e as outras 20. Só essa coluna valia tanto como 77 features de áudio.

Corrigi os dois, medi por paciente com validação cruzada e intervalos de confiança, e o resultado honesto é este: recall de COPD 1,00, recall de URTI 0,24. Saudável e URTI soam igual no sinal (AUC 0,45).

A decisão mais importante não foi o algoritmo. Foi fixar o limiar para nunca dar alta a um doente, e assumir o custo em falsos positivos.

Código, métricas e testes no GitHub: github.com/pintoleodgar/projeto-cdg-respiratori

#MachineLearning #HealthTech #DataScience #Portfolio
