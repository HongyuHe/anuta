#!/usr/bin/env bash

# Read input PCAP file
# -r "$1" : Input pcap file

# Output as CSV (fields mode)
# -T fields : Output only selected fields
# -E separator=, : Use comma as CSV delimiter
# -E quote=d : Use double quotes around each field
# -E header=y : Include column headers in output

# === Frame-level metadata ===
# frame.number : Packet number in capture file
# frame.time_epoch : Timestamp in UNIX epoch (float)
# frame.len : Full length of the packet on the wire
# frame.cap_len : Length of captured portion of the packet
# frame.encap_type : Link-layer type (e.g., Ethernet)

# === Ethernet (L2) ===
# eth.src : Source MAC address
# eth.dst : Destination MAC address

# === IPv4 header fields ===
# ip.version : IP version (should be 4)
# ip.hdr_len : IP header length in bytes
# ip.len : Total length of the IP packet
# ip.id : IP identification field (used for fragmentation)
# ip.flags : Fragmentation flags (DF, MF)
# ip.frag_offset : Fragment offset value
# ip.ttl : Time To Live value
# ip.checksum : Header checksum for error checking
# ip.proto : Protocol number (e.g., 6=TCP, 17=UDP, 1=ICMP)
# ip.src : Source IP address
# ip.dst : Destination IP address

# === TCP fields (if applicable) ===
# tcp.srcport : Source TCP port
# tcp.dstport : Destination TCP port
# tcp.hdr_len : TCP header length in bytes
# tcp.flags : TCP flags in hex (SYN, ACK, etc.)
# tcp.len : TCP payload size (excluding header)
# tcp.seq : Sequence number
# tcp.ack : Acknowledgment number
# tcp.urgent_pointer : Urgent pointer (rarely used)
# tcp.window_size_value : Advertised window size
# tcp.checksum : TCP checksum
# tcp.options : Full TCP options string
# tcp.options.timestamp.tsval : TCP timestamp value (TSval)
# tcp.options.timestamp.tsecr : TCP timestamp echo reply (TSecr)

# === UDP fields (if applicable) ===
# udp.srcport : Source UDP port
# udp.dstport : Destination UDP port
# udp.length : UDP datagram size (header + payload)
# udp.checksum : UDP checksum

# === Display protocol label ===
# _ws.col.Protocol : Wireshark column for protocol label

# === Output to file ===
# "$2" : Output CSV file

tshark -r "$1" -T fields -E separator=, -E quote=d -E header=y \
  -e frame.number \
  -e frame.time_epoch \
  -e frame.len \
  -e frame.cap_len \
  -e frame.encap_type \
  -e eth.src \
  -e eth.dst \
  -e ip.version \
  -e ip.hdr_len \
  -e ip.len \
  -e ip.id \
  -e ip.flags \
  -e ip.frag_offset \
  -e ip.ttl \
  -e ip.checksum \
  -e ip.proto \
  -e ip.src \
  -e ip.dst \
  -e tcp.srcport \
  -e tcp.dstport \
  -e tcp.hdr_len \
  -e tcp.flags \
  -e tcp.len \
  -e tcp.seq \
  -e tcp.ack \
  -e tcp.urgent_pointer \
  -e tcp.window_size_value \
  -e tcp.checksum \
  -e tcp.options \
  -e tcp.options.timestamp.tsval \
  -e tcp.options.timestamp.tsecr \
  -e udp.srcport \
  -e udp.dstport \
  -e udp.length \
  -e udp.checksum \
  -e _ws.col.Protocol \
  > "$2"