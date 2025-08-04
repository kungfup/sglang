#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <memory>
#include <queue>
#include "mscclpp_allreduce.cuh"

namespace semi_pd {

// Semi-PD 异步内存句柄管理器
class AsyncMemoryHandleManager {
private:
    std::unordered_map<void*, std::shared_ptr<mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle>>> handle_cache_;
    std::unordered_map<void*, std::unordered_map<int, mscclpp::MemoryChannel>> memory_channels_cache_;
    std::queue<cudaEvent_t> event_pool_;
    cudaStream_t async_stream_;
    bool initialized_;

public:
    AsyncMemoryHandleManager() : initialized_(false) {
        // 创建专用的异步流
        CHECK_CUDA_SUCCESS(cudaStreamCreate(&async_stream_));
        
        // 预分配事件池
        for (int i = 0; i < 16; ++i) {
            cudaEvent_t event;
            CHECK_CUDA_SUCCESS(cudaEventCreate(&event));
            event_pool_.push(event);
        }
        initialized_ = true;
    }

    ~AsyncMemoryHandleManager() {
        if (initialized_) {
            CHECK_CUDA_SUCCESS(cudaStreamDestroy(async_stream_));
            while (!event_pool_.empty()) {
                CHECK_CUDA_SUCCESS(cudaEventDestroy(event_pool_.front()));
                event_pool_.pop();
            }
        }
    }

    // 异步获取内存句柄（无同步）
    mscclpp::MemoryChannelDeviceHandle* getMemoryHandleAsync(
        void* input_ptr,
        mscclpp::Msccl1NodeLLcontext* context,
        cudaStream_t user_stream,
        cudaEvent_t* completion_event = nullptr) {
        
        auto it = handle_cache_.find(input_ptr);
        if (it != handle_cache_.end()) {
            // 缓存命中，直接返回
            return it->second->data();
        }

        // 缓存未命中，异步创建新句柄
        auto device_handle = std::make_shared<mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle>>(
            context->comm_group_->world_size_ - 1);
        std::unordered_map<int, mscclpp::MemoryChannel> memory_channels;

        // 在异步流中创建内存句柄
        context->comm_group_->make_device_memory_handle_base_on_new_ptr(
            context->memory_channels_,
            context->registered_sm_memories_,
            context->memory_semaphores_,
            memory_channels,
            *device_handle,
            input_ptr,
            context->scratch_,
            async_stream_);  // 使用专用异步流

        // 创建完成事件
        cudaEvent_t event;
        if (!event_pool_.empty()) {
            event = event_pool_.front();
            event_pool_.pop();
        } else {
            CHECK_CUDA_SUCCESS(cudaEventCreate(&event));
        }

        // 记录异步操作完成点
        CHECK_CUDA_SUCCESS(cudaEventRecord(event, async_stream_));
        
        // 在用户流中等待异步操作完成（非阻塞）
        CHECK_CUDA_SUCCESS(cudaStreamWaitEvent(user_stream, event, 0));

        // 缓存结果
        handle_cache_[input_ptr] = device_handle;
        memory_channels_cache_[input_ptr] = std::move(memory_channels);

        // 返回完成事件（如果需要）
        if (completion_event) {
            *completion_event = event;
        } else {
            event_pool_.push(event);  // 回收事件
        }

        return device_handle->data();
    }

    // 类似的方法用于 2-node context
    mscclpp::MemoryChannelDeviceHandle* getMemoryHandleAsync(
        void* input_ptr,
        mscclpp::Msccl2NodeLLcontext* context,
        cudaStream_t user_stream,
        cudaEvent_t* completion_event = nullptr) {
        
        auto it = handle_cache_.find(input_ptr);
        if (it != handle_cache_.end()) {
            return it->second->data();
        }

        auto device_handle = std::make_shared<mscclpp::GpuBuffer<mscclpp::MemoryChannelDeviceHandle>>(7);
        std::unordered_map<int, mscclpp::MemoryChannel> memory_channels;

        context->comm_group_->make_device_memory_handle_base_on_new_ptr(
            context->memory_channels_,
            context->registered_sm_memories_,
            context->memory_semaphores_,
            memory_channels,
            *device_handle,
            input_ptr,
            context->scratch_,
            async_stream_);

        cudaEvent_t event;
        if (!event_pool_.empty()) {
            event = event_pool_.front();
            event_pool_.pop();
        } else {
            CHECK_CUDA_SUCCESS(cudaEventCreate(&event));
        }

        CHECK_CUDA_SUCCESS(cudaEventRecord(event, async_stream_));
        CHECK_CUDA_SUCCESS(cudaStreamWaitEvent(user_stream, event, 0));

        handle_cache_[input_ptr] = device_handle;
        memory_channels_cache_[input_ptr] = std::move(memory_channels);

        if (completion_event) {
            *completion_event = event;
        } else {
            event_pool_.push(event);
        }

        return device_handle->data();
    }

    void clearCache() {
        handle_cache_.clear();
        memory_channels_cache_.clear();
    }
};

// 全局单例管理器
inline AsyncMemoryHandleManager& getAsyncMemoryHandleManager() {
    static AsyncMemoryHandleManager instance;
    return instance;
}

} // namespace semi_pd 