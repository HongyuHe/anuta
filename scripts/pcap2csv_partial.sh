#!/usr/bin/env bash

#* Field: Description
#* frame.number: Packet number (No. in Wireshark).
#* frame.time: Timestamp of the packet capture (Time).
#* ip.src: Source IP address (Source).
#* ip.dst: Destination IP address (Destination).
#* _ws.col.Protocol: Protocol used (Protocol).
#* frame.len: Total length of the frame (Length).
#* tcp.srcport: Source port number.
#* tcp.dstport: Destination port number.
#* tcp.flags: TCP flags as a bitmask.
#* _ws.col.Info: Summary information (e.g., [ACK]).
#* tcp.len: Length of the TCP segment.
#* tcp.seq: Sequence number.
#* tcp.ack: Acknowledgment number.
#* tcp.window_size_value: TCP window size.
#* tcp.options.timestamp.tsval: TCP timestamp value.
#* tcp.options.timestamp.tsecr: TCP timestamp echo reply

tshark -r "$1" -T fields -E separator=, -E quote=d -E header=y \
    -e frame.number \
    -e frame.time \
    -e ip.src \
    -e ip.dst \
    -e _ws.col.Protocol \
    -e frame.len \
    -e tcp.srcport \
    -e tcp.dstport \
    -e udp.srcport \
    -e udp.dstport \
    -e tcp.flags \
    -e tcp.len \
    -e udp.length \
    -e tcp.seq \
    -e tcp.ack \
    -e tcp.window_size_value \
    -e tcp.options.timestamp.tsval \
    -e tcp.options.timestamp.tsecr \
    > "$2"
    # -e _ws.col.Info \
