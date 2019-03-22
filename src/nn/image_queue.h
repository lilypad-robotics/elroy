#ifndef IMAGEQUEUE_H
#define IMAGEQUEUE_H

#include <mutex>
#include <condition_variable>
#include <queue>

template<typename T>
class ConsumerProducerQueue
{

public:
    ConsumerProducerQueue(int mxsz,bool dropFrame) :
            maxSize(mxsz), dropFrame(dropFrame)
    { }
    void add(T request);
    void consume(T &request);
    bool isFull() const;
    bool isEmpty() const;
    int length() const;
    void clear();
private:
    std::condition_variable cond;
    std::mutex mutex;
    std::queue<T> cpq;
    int maxSize;
    bool dropFrame;
};

#endif
