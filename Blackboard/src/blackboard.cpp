/*--------------------------------------------------------------------

******************************************************************************
* @file blackboard.cpp
* @author Isaac Jesus da Silva - ROBOFEI-HT - FEI
* @version V0.0.3
* @created 07/04/2014
* @Modified 10/09/2015
* @e-mail isaac25silva@yahoo.com.br
* @brief blackboard
****************************************************************************

Arquivo fonte contendo as funções que cria ou acopla a memória compartilhada

/--------------------------------------------------------------------*/

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdint.h>
#include "blackboard.h"

//#define DEBUG

//#define KEY 123   //we are using parameters allowing us to create different blackboards to each robot

int *mem ; //Variável que manipula memória compartilhada
float *memf ; //Variável que manipula memória compartilhada

//Depois de criado a memória compartilhada, para verificar se ela realmente foi criada
// e quantos processos estão utilizando, digite no terminal o comando $ipcs -m
// será a memoria criada ->   key = 0x0000007b    bytes = 2048
// nattch = number of attached processes

void write_int(int *Mem, int index, int valor)
{
    *(Mem+index) = valor;
}

void write_float(int *Mem, int index, float valor)
{
    float* Memf;
    Memf = (float*)(Mem+125);
    *(Memf+index) = valor;
}

int read_int(int *Mem, int index)
{
    return *(Mem + index);
}

float read_float(int *Mem, int index)
{
    float* Memf;
    Memf = (float*)(Mem+125);
    return *(Memf + index);
}

int* using_shared_memory(int KEY)
{
    // --- Variaveis usada para memoria compartilhada -----
    int shmid ; // identificador da memoria comum //
    const unsigned short int size = 2048; // tamanho da memória em Bytes
    int flag = 0;
    //-----------------------------------------------------

     // Recuperando ou criando uma memoria compartilhada-------------------
     //

     //shmget:para criar um segmento de memória compartilhada
     if (( shmid = shmget((key_t)KEY, size,0)) == -1)
     {
          perror("shmget error") ;
          printf("\n Memory will be created \n");
         //return(1) ;
        if (( shmid = shmget((key_t)KEY, size, IPC_CREAT|IPC_EXCL|SHM_R|SHM_W)) == -1)
        {
            perror("shmget error") ;
            //return(1) ;
        }

     }
#ifdef DEBUG
     printf("PID: %d \n",getpid()) ;
     printf("Segment identifier: %d \n",shmid) ;
     printf("Segment is associated to the unique key: %d\n",(key_t)KEY);
#endif
    //
    // acoplamento do processo a zona de memoria
    //recuperacao do pornteiro sobre a area de memoria comum
    //
    //shmat:retorna um pointeiro para o segmento de memória compartilhada
     if ((mem = (int*)shmat (shmid, 0, flag)) == (int*)-1){
          perror("Impossible linkage!") ;
          //return (2) ;
     }

     memf = (float*)(mem+125);
     //---------------------------------------------------------------------

            /* destruicao do segmento */
            //if ((shmctl(shmid, IPC_RMID, NULL)) == -1){
            // perror("Erro shmctl()");
             // return(1) ;
            //}
    
    printf("%d\n",(*mem));
    return(mem); 

}
