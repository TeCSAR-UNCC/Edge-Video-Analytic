#ifndef __REID_COMMS_HPP__
#define __REID_COMMS_HPP__

/*************************************/
/*          ADDING COMMS             */
/*************************************/
#define sPORT 76543
#define rPORT 76544
#define lIPADDR INADDR_ANY
#define NUM_THREADS 2

pthread_mutex_t locks[NUM_THREADS];
  
int sendLock = 0;
int recvLock = 1;

queue<personType> sendQ;
queue<reIDType> recvQ;

char * IPADDR;
/**************************************/

void *sendData(void*) {
    /* Initializing network parameters */
    int sock;
    //char* IPADDR = "192.168.0.26";
    struct sockaddr_in servaddr;

    sock = socket(AF_INET,SOCK_STREAM,0);
    bzero(&servaddr,sizeof servaddr);

    servaddr.sin_family=AF_INET;
    servaddr.sin_port=htons(sPORT);

    inet_pton(AF_INET, IPADDR, &(servaddr.sin_addr));

    connect(sock,(struct sockaddr *)&servaddr,sizeof(servaddr)); 

    personType data;
    std::cout << "Entering Send function." << std::endl;

    while(1) {
        pthread_mutex_lock(&locks[sendLock]);
        if(!sendQ.empty()) {
            std::cout << "Inside send Queue\n";
            data = sendQ.front();
            sendQ.pop();
            pthread_mutex_unlock(&locks[sendLock]);
            cout << "data.label = " << data.label << endl;

            int status = send(sock,reinterpret_cast<const char*> (&data), sizeof(personType), 0);
            cout << "Sent Data: " << status << endl;
        }
        else{
            pthread_mutex_unlock(&locks[sendLock]);
        }
    }
    close(sock);
}

void *recvData(void*) {
    while(1) {
        int listen_fd, comm_fd;

        struct sockaddr_in servaddr;

        listen_fd = socket(AF_INET, SOCK_STREAM, 0);

        bzero( &servaddr, sizeof(servaddr));

        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = htons(lIPADDR);
        servaddr.sin_port = htons(rPORT);

        bind(listen_fd, (struct sockaddr *) &servaddr, sizeof(servaddr));

        listen(listen_fd, 10);
        comm_fd = accept(listen_fd, (struct sockaddr*) NULL, NULL);

        reIDType data;
        int status = 1;

        while(status > 0) {
            status = recv(comm_fd, &data, sizeof(reIDType), 0);

            pthread_mutex_lock(&locks[recvLock]);
            recvQ.push(data);
	            cout << "data.newID = " << data.newID << endl;
            pthread_mutex_unlock(&locks[recvLock]);

            cout << " status: " << status << endl;
        }
        close(comm_fd);
        close(listen_fd);
    }
}

#endif //__REID_COMMS_HPP__
