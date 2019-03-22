#include "image_queue.h"

template <typename T>
void ConsumerProducerQueue<T>::add(T request)
{
    std::unique_lock<std::mutex> lock(mutex);
    if(dropFrame && isFull())
    {
        lock.unlock();
        return;
    }
    else {
        cond.wait(lock, [this]() { return !isFull(); });
        cpq.push(request);
        cond.notify_all();
    }
}

template <typename T>
void ConsumerProducerQueue<T>::consume(T &request)
{
    std::unique_lock<std::mutex> lock(mutex);
    cond.wait(lock, [this]()
    { return !isEmpty(); });
    request = cpq.front();
    cpq.pop();
    cond.notify_all();
}

template <typename T>
void ConsumerProducerQueue<T>::clear()
{
    std::unique_lock<std::mutex> lock(mutex);
    while (!isEmpty())
    {
        cpq.pop();
    }
    lock.unlock();
    cond.notify_all();
}

template <typename T>
bool ConsumerProducerQueue<T>::isFull() const
{
    return cpq.size() >= maxSize;
}

template <typename T>
bool ConsumerProducerQueue<T>::isEmpty() const
{
    return cpq.size() == 0;
}

template <typename T>
int ConsumerProducerQueue<T>::length() const
{
        return cpq.size();
}
