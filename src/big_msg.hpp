#pragma once

// Charm++ marshalled messages hold less than 2 GiB (int sizes in charmxi, CmiAlloc and the
// envelope): larger byte payloads go as fragments (receive_frag), which the receiving chare
// reassembles before calling the entry the payload was meant for.

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "charm++.h"

enum MsgKind : int {
    MSG_DATA = 0,           // receive_data (host payload); meta: input_index, name, ndims, region
    MSG_STAGED = 1,         // receive_staged; meta as MSG_DATA
    MSG_REDUCE_PARTIAL = 2, // receive_reduce_partial
    MSG_LINALG = 3,         // receive_linalg
    MSG_INDEX = 4,          // receive_index_msg
    MSG_SHAPE = 5,          // receive_shape
    MSG_UPLOAD_TILE = 6,    // receive_upload_tile; meta: name
};

/// Largest payload sent as one message (CNUM_MSG_MAX_BYTES overrides it, for tests).
int64_t ct_msg_max_bytes();

template <typename P>
void ct_send_frags(P&& proxy, int kind, int node_id, const std::vector<int>& meta, int64_t size,
                   const char* data) {
    static thread_local int64_t seq = 0;
    ++seq;
    int64_t lim = ct_msg_max_bytes();
    for (int64_t off = 0; off < size; off += lim) {
        int64_t n = std::min(lim, size - off);
        proxy.receive_frag(node_id, kind, CkMyPe(), seq, (int)meta.size(), meta.data(), size, off,
                           n, data + off);
    }
}

/// Sends a byte payload to the receive_* entry of `kind` (MSG_REDUCE_PARTIAL..MSG_UPLOAD_TILE).
template <typename P>
void ct_send_bytes(P&& proxy, MsgKind kind, int node_id, int64_t size, const char* data,
                   int name = 0) {
    if (size > ct_msg_max_bytes()) {
        ct_send_frags(proxy, kind, node_id, {name}, size, data);
        return;
    }
    switch (kind) {
    case MSG_REDUCE_PARTIAL: proxy.receive_reduce_partial(node_id, size, data); break;
    case MSG_LINALG: proxy.receive_linalg(node_id, size, data); break;
    case MSG_INDEX: proxy.receive_index_msg(node_id, size, data); break;
    case MSG_SHAPE: proxy.receive_shape(node_id, size, data); break;
    case MSG_UPLOAD_TILE: proxy.receive_upload_tile(node_id, name, size, data); break;
    default: CkAbort("ct_send_bytes: bad message kind %d", (int)kind);
    }
}

inline std::vector<int> ct_data_meta(int input_index, int name, int ndims,
                                     const int* region_data) {
    std::vector<int> meta = {input_index, name, ndims};
    meta.insert(meta.end(), region_data, region_data + ndims * 3);
    return meta;
}

#ifndef USE_KOKKOS
/// receive_data with a host payload.
template <typename P>
void ct_send_data(P&& proxy, int node_id, int input_index, int name, int ndims,
                  const int* region_data, int64_t size, const char* data) {
    if (size > ct_msg_max_bytes())
        ct_send_frags(proxy, MSG_DATA, node_id, ct_data_meta(input_index, name, ndims, region_data),
                      size, data);
    else
        proxy.receive_data(node_id, input_index, name, ndims, region_data, size, data);
}
#endif
