Eq(TcpFlags1, 0) >> Eq(TcpFlags2, 1)
Eq(TcpFlags1, 0) >> Eq(TcpAck2, TcpSeq1+1)
Eq(TcpFlags1, 0) >> Eq(TcpSeq3, TcpSeq1+1)
Eq(TcpFlags1, 0) >> Eq(TcpAck3, TcpSeq2+1)
Eq(TcpFlags1, 0) >> Eq(TcpFlags3, 2)
Eq(TcpFlags1, 3) >> (TcpLen1 > 0)
Eq(TcpFlags2, 3) >> (TcpLen2 > 0)
Eq(TcpFlags3, 3) >> (TcpLen3 > 0)
Eq(IpSrc1, 0) >> (Eq(TcpDstport1, 1) & Eq(TcpSrcport1, 0))
Eq(IpSrc2, 0) >> (Eq(TcpDstport2, 1) & Eq(TcpSrcport2, 0))
Eq(IpSrc3, 0) >> (Eq(TcpDstport3, 1) & Eq(TcpSrcport3, 0))
Eq(IpSrc1, 1) >> (Eq(TcpDstport1, 0) & Eq(TcpSrcport1, 1))
Eq(IpSrc2, 1) >> (Eq(TcpDstport2, 0) & Eq(TcpSrcport2, 1))
Eq(IpSrc3, 1) >> (Eq(TcpDstport3, 0) & Eq(TcpSrcport3, 1))