# AMD RDNA2 Instruction Set - Navi 24 (RX 6400/6500 XT/6500M)

Complete instruction listing for AMD Radeon RX 6400/6500 XT/6500M GPUs (Navi 24 architecture, RDNA2 generation).

## SCALAR ALU INSTRUCTIONS (SOP2)
32-bit and 64-bit integer operations on scalar registers (SGPRs).

### Integer Arithmetic
- **S_ADD_U32** - Add unsigned 32-bit integers
- **S_ADD_I32** - Add signed 32-bit integers  
- **S_ADDC_U32** - Add with carry
- **S_SUB_U32** - Subtract unsigned 32-bit integers
- **S_SUB_I32** - Subtract signed 32-bit integers
- **S_SUBB_U32** - Subtract with borrow
- **S_MIN_I32** - Minimum signed 32-bit
- **S_MIN_U32** - Minimum unsigned 32-bit
- **S_MAX_I32** - Maximum signed 32-bit
- **S_MAX_U32** - Maximum unsigned 32-bit
- **S_MUL_I32** - Multiply signed 32-bit
- **S_MUL_HI_U32** - Multiply high unsigned 32-bit
- **S_MUL_HI_I32** - Multiply high signed 32-bit
- **S_ABSDIFF_I32** - Absolute difference signed 32-bit
- **S_LSHL1_ADD_U32** - Left shift by 1 and add
- **S_LSHL2_ADD_U32** - Left shift by 2 and add
- **S_LSHL3_ADD_U32** - Left shift by 3 and add
- **S_LSHL4_ADD_U32** - Left shift by 4 and add

### Bitwise Operations
- **S_AND_B32** - Bitwise AND 32-bit
- **S_AND_B64** - Bitwise AND 64-bit
- **S_OR_B32** - Bitwise OR 32-bit
- **S_OR_B64** - Bitwise OR 64-bit
- **S_XOR_B32** - Bitwise XOR 32-bit
- **S_XOR_B64** - Bitwise XOR 64-bit
- **S_ANDN2_B32** - AND NOT 32-bit (S0 & ~S1)
- **S_ANDN2_B64** - AND NOT 64-bit
- **S_ORN2_B32** - OR NOT 32-bit (S0 | ~S1)
- **S_ORN2_B64** - OR NOT 64-bit
- **S_NAND_B32** - NAND 32-bit
- **S_NAND_B64** - NAND 64-bit
- **S_NOR_B32** - NOR 32-bit
- **S_NOR_B64** - NOR 64-bit
- **S_XNOR_B32** - XNOR 32-bit
- **S_XNOR_B64** - XNOR 64-bit

### Shift Operations
- **S_LSHL_B32** - Logical shift left 32-bit
- **S_LSHL_B64** - Logical shift left 64-bit
- **S_LSHR_B32** - Logical shift right 32-bit
- **S_LSHR_B64** - Logical shift right 64-bit
- **S_ASHR_I32** - Arithmetic shift right 32-bit
- **S_ASHR_I64** - Arithmetic shift right 64-bit

### Bit Field Operations
- **S_BFM_B32** - Bit field mask 32-bit
- **S_BFM_B64** - Bit field mask 64-bit
- **S_BFE_U32** - Bit field extract unsigned 32-bit
- **S_BFE_I32** - Bit field extract signed 32-bit
- **S_BFE_U64** - Bit field extract unsigned 64-bit
- **S_BFE_I64** - Bit field extract signed 64-bit

### Packing Operations
- **S_PACK_LL_B32_B16** - Pack two low 16-bit values into 32-bit
- **S_PACK_LH_B32_B16** - Pack low-high 16-bit values
- **S_PACK_HH_B32_B16** - Pack two high 16-bit values

### Conditional Select
- **S_CSELECT_B32** - Conditional select 32-bit
- **S_CSELECT_B64** - Conditional select 64-bit

### Compare with Constant
- **S_CMPK_EQ_I32** - Compare equal signed
- **S_CMPK_LG_I32** - Compare not equal signed
- **S_CMPK_GT_I32** - Compare greater than signed
- **S_CMPK_GE_I32** - Compare greater or equal signed
- **S_CMPK_LT_I32** - Compare less than signed
- **S_CMPK_LE_I32** - Compare less or equal signed
- **S_CMPK_EQ_U32** - Compare equal unsigned
- **S_CMPK_LG_U32** - Compare not equal unsigned
- **S_CMPK_GT_U32** - Compare greater than unsigned
- **S_CMPK_GE_U32** - Compare greater or equal unsigned
- **S_CMPK_LT_U32** - Compare less than unsigned
- **S_CMPK_LE_U32** - Compare less or equal unsigned

## SCALAR ALU INSTRUCTIONS (SOP1)
Single operand scalar operations.

### Move and Convert
- **S_MOV_B32** - Move 32-bit
- **S_MOV_B64** - Move 64-bit
- **S_CMOV_B32** - Conditional move 32-bit
- **S_CMOV_B64** - Conditional move 64-bit
- **S_MOVK_I32** - Move constant 32-bit
- **S_CMOVK_I32** - Conditional move constant

### Bitwise Operations
- **S_NOT_B32** - Bitwise NOT 32-bit
- **S_NOT_B64** - Bitwise NOT 64-bit
- **S_BREV_B32** - Bit reverse 32-bit
- **S_BREV_B64** - Bit reverse 64-bit
- **S_WQM_B32** - Whole quad mode 32-bit
- **S_WQM_B64** - Whole quad mode 64-bit

### Bit Counting
- **S_BCNT0_I32_B32** - Count zero bits 32-bit
- **S_BCNT0_I32_B64** - Count zero bits 64-bit
- **S_BCNT1_I32_B32** - Count one bits 32-bit
- **S_BCNT1_I32_B64** - Count one bits 64-bit

### Find First Bit
- **S_FF0_I32_B32** - Find first zero 32-bit
- **S_FF0_I32_B64** - Find first zero 64-bit
- **S_FF1_I32_B32** - Find first one 32-bit
- **S_FF1_I32_B64** - Find first one 64-bit
- **S_FLBIT_I32_B32** - Find last bit 32-bit unsigned
- **S_FLBIT_I32_B64** - Find last bit 64-bit unsigned
- **S_FLBIT_I32** - Find last bit signed 32-bit
- **S_FLBIT_I32_I64** - Find last bit signed 64-bit

### Sign Extension
- **S_SEXT_I32_I8** - Sign extend 8-bit to 32-bit
- **S_SEXT_I32_I16** - Sign extend 16-bit to 32-bit

### Bit Set/Clear
- **S_BITSET0_B32** - Clear bit 32-bit
- **S_BITSET0_B64** - Clear bit 64-bit
- **S_BITSET1_B32** - Set bit 32-bit
- **S_BITSET1_B64** - Set bit 64-bit

### Quad Operations
- **S_QUADMASK_B32** - Generate quad mask 32-bit
- **S_QUADMASK_B64** - Generate quad mask 64-bit

### Relative Addressing
- **S_MOVRELS_B32** - Move relative source 32-bit
- **S_MOVRELS_B64** - Move relative source 64-bit
- **S_MOVRELD_B32** - Move relative destination 32-bit
- **S_MOVRELD_B64** - Move relative destination 64-bit

### Absolute Value
- **S_ABS_I32** - Absolute value 32-bit

### EXEC Mask Operations
- **S_AND_SAVEEXEC_B32/B64** - AND with EXEC and save
- **S_OR_SAVEEXEC_B32/B64** - OR with EXEC and save
- **S_XOR_SAVEEXEC_B32/B64** - XOR with EXEC and save
- **S_ANDN2_SAVEEXEC_B32/B64** - ANDN2 with EXEC and save
- **S_ORN2_SAVEEXEC_B32/B64** - ORN2 with EXEC and save
- **S_NAND_SAVEEXEC_B32/B64** - NAND with EXEC and save
- **S_NOR_SAVEEXEC_B32/B64** - NOR with EXEC and save
- **S_XNOR_SAVEEXEC_B32/B64** - XNOR with EXEC and save
- **S_ANDN1_SAVEEXEC_B32/B64** - ANDN1 with EXEC and save
- **S_ORN1_SAVEEXEC_B32/B64** - ORN1 with EXEC and save
- **S_ANDN1_WREXEC_B32/B64** - ANDN1 write EXEC
- **S_ANDN2_WREXEC_B32/B64** - ANDN2 write EXEC

### Control Flow
- **S_GETPC_B64** - Get program counter
- **S_SETPC_B64** - Set program counter
- **S_SWAPPC_B64** - Swap program counter
- **S_RFE_B64** - Return from exception
- **S_CALL_B64** - Call subroutine

### System
- **S_VERSION** - Get ISA version
- **S_GETREG_B32** - Get hardware register
- **S_SETREG_B32** - Set hardware register
- **S_SETREG_IMM32_B32** - Set hardware register immediate

### Synchronization
- **S_WAITCNT_VSCNT** - Wait for vector store count
- **S_WAITCNT_VMCNT** - Wait for vector memory count
- **S_WAITCNT_EXPCNT** - Wait for export count
- **S_WAITCNT_LGKMCNT** - Wait for LGK memory count

### Subvector Loops
- **S_SUBVECTOR_LOOP_BEGIN** - Begin subvector loop
- **S_SUBVECTOR_LOOP_END** - End subvector loop

### Miscellaneous
- **S_ADDK_I32** - Add constant
- **S_MULK_I32** - Multiply constant

## VECTOR ALU INSTRUCTIONS (VOP2)
Two-operand vector operations (32 lanes per wave).

### Integer Arithmetic
- **V_ADD_NC_U32** - Add unsigned no carry
- **V_SUB_NC_U32** - Subtract unsigned no carry
- **V_SUBREV_NC_U32** - Reverse subtract unsigned no carry
- **V_ADD_CO_CI_U32** - Add with carry in/out
- **V_SUB_CO_CI_U32** - Subtract with carry in/out
- **V_SUBREV_CO_CI_U32** - Reverse subtract with carry in/out
- **V_MIN_I32** - Minimum signed 32-bit
- **V_MIN_U32** - Minimum unsigned 32-bit
- **V_MAX_I32** - Maximum signed 32-bit
- **V_MAX_U32** - Maximum unsigned 32-bit
- **V_MUL_I32_I24** - Multiply signed 24-bit
- **V_MUL_U32_U24** - Multiply unsigned 24-bit
- **V_MUL_HI_I32_I24** - Multiply high signed 24-bit
- **V_MUL_HI_U32_U24** - Multiply high unsigned 24-bit

### Float Arithmetic (FP32)
- **V_ADD_F32** - Add single precision
- **V_SUB_F32** - Subtract single precision
- **V_SUBREV_F32** - Reverse subtract single precision
- **V_MUL_F32** - Multiply single precision
- **V_MUL_LEGACY_F32** - Legacy multiply (denorm support)
- **V_MAX_F32** - Maximum single precision
- **V_MIN_F32** - Minimum single precision
- **V_FMAC_F32** - Fused multiply-add (FMA)
- **V_FMAC_LEGACY_F32** - Legacy FMA
- **V_FMAAK_F32** - FMA with inline constant
- **V_FMAMK_F32** - FMA with multiplied constant
- **V_LDEXP_F32** - Load exponent

### Float Arithmetic (FP16)
- **V_ADD_F16** - Add half precision
- **V_SUB_F16** - Subtract half precision
- **V_SUBREV_F16** - Reverse subtract half precision
- **V_MUL_F16** - Multiply half precision
- **V_MAX_F16** - Maximum half precision
- **V_MIN_F16** - Minimum half precision
- **V_FMAC_F16** - FMA half precision
- **V_FMAAK_F16** - FMA half with inline constant
- **V_FMAMK_F16** - FMA half with multiplied constant
- **V_LDEXP_F16** - Load exponent half precision

### Bitwise Operations
- **V_AND_B32** - Bitwise AND
- **V_OR_B32** - Bitwise OR
- **V_XOR_B32** - Bitwise XOR
- **V_XNOR_B32** - Bitwise XNOR

### Shift Operations
- **V_LSHLREV_B32** - Logical shift left reversed
- **V_LSHRREV_B32** - Logical shift right reversed
- **V_ASHRREV_I32** - Arithmetic shift right reversed

### Conditional
- **V_CNDMASK_B32** - Conditional mask

### Packing
- **V_CVT_PKRTZ_F16_F32** - Pack two FP32 to FP16 with round-to-zero
- **V_PK_FMAC_F16** - Packed half precision FMA

### Dot Products
- **V_DOT2C_F32_F16** - Dot product 2 FP16 accumulate FP32
- **V_DOT4C_I32_I8** - Dot product 4 INT8 accumulate INT32

## VECTOR ALU INSTRUCTIONS (VOP1)
Single-operand vector operations.

### Move
- **V_MOV_B32** - Move 32-bit
- **V_NOP** - No operation
- **V_PIPEFLUSH** - Pipeline flush
- **V_SWAP_B32** - Swap two VGPRs
- **V_SWAPREL_B32** - Swap relative

### Bitwise
- **V_NOT_B32** - Bitwise NOT
- **V_BFREV_B32** - Bit reverse

### Float Conversion
- **V_CVT_F32_F16** - Convert FP16 to FP32
- **V_CVT_F16_F32** - Convert FP32 to FP16
- **V_CVT_F64_F32** - Convert FP32 to FP64
- **V_CVT_F32_F64** - Convert FP64 to FP32

### Integer to Float
- **V_CVT_F32_I32** - Convert INT32 to FP32
- **V_CVT_F32_U32** - Convert UINT32 to FP32
- **V_CVT_F64_I32** - Convert INT32 to FP64
- **V_CVT_F64_U32** - Convert UINT32 to FP64
- **V_CVT_F16_I16** - Convert INT16 to FP16
- **V_CVT_F16_U16** - Convert UINT16 to FP16
- **V_CVT_F32_UBYTE0** - Convert unsigned byte 0 to FP32
- **V_CVT_F32_UBYTE1** - Convert unsigned byte 1 to FP32
- **V_CVT_F32_UBYTE2** - Convert unsigned byte 2 to FP32
- **V_CVT_F32_UBYTE3** - Convert unsigned byte 3 to FP32

### Float to Integer
- **V_CVT_I32_F32** - Convert FP32 to INT32
- **V_CVT_U32_F32** - Convert FP32 to UINT32
- **V_CVT_I32_F64** - Convert FP64 to INT32
- **V_CVT_U32_F64** - Convert FP64 to UINT32
- **V_CVT_I16_F16** - Convert FP16 to INT16
- **V_CVT_U16_F16** - Convert FP16 to UINT16
- **V_CVT_RPI_I32_F32** - Round to positive infinity then convert
- **V_CVT_FLR_I32_F32** - Floor then convert
- **V_CVT_OFF_F32_I4** - Offset convert

### Normalized Conversions
- **V_CVT_NORM_I16_F16** - Convert FP16 to normalized INT16
- **V_CVT_NORM_U16_F16** - Convert FP16 to normalized UINT16

### Rounding
- **V_CEIL_F16** - Ceiling half precision
- **V_CEIL_F32** - Ceiling single precision
- **V_CEIL_F64** - Ceiling double precision
- **V_FLOOR_F16** - Floor half precision
- **V_FLOOR_F32** - Floor single precision
- **V_FLOOR_F64** - Floor double precision
- **V_TRUNC_F16** - Truncate half precision
- **V_TRUNC_F32** - Truncate single precision
- **V_TRUNC_F64** - Truncate double precision
- **V_RNDNE_F16** - Round to nearest even half
- **V_RNDNE_F32** - Round to nearest even single
- **V_RNDNE_F64** - Round to nearest even double
- **V_FRACT_F16** - Fractional part half
- **V_FRACT_F32** - Fractional part single
- **V_FRACT_F64** - Fractional part double

### Transcendental (FP32)
- **V_SIN_F32** - Sine
- **V_COS_F32** - Cosine
- **V_EXP_F32** - Exponential (2^x)
- **V_LOG_F32** - Logarithm (log2)
- **V_SQRT_F32** - Square root
- **V_RCP_F32** - Reciprocal (1/x)
- **V_RSQ_F32** - Reciprocal square root
- **V_RCP_IFLAG_F32** - Reciprocal with integer flag

### Transcendental (FP16)
- **V_SIN_F16** - Sine half
- **V_COS_F16** - Cosine half
- **V_EXP_F16** - Exponential half
- **V_LOG_F16** - Logarithm half
- **V_SQRT_F16** - Square root half
- **V_RCP_F16** - Reciprocal half
- **V_RSQ_F16** - Reciprocal square root half

### Transcendental (FP64)
- **V_SQRT_F64** - Square root double
- **V_RCP_F64** - Reciprocal double
- **V_RSQ_F64** - Reciprocal square root double

### Float Manipulation
- **V_FREXP_MANT_F16** - Extract mantissa half
- **V_FREXP_MANT_F32** - Extract mantissa single
- **V_FREXP_MANT_F64** - Extract mantissa double
- **V_FREXP_EXP_I16_F16** - Extract exponent half
- **V_FREXP_EXP_I32_F32** - Extract exponent single
- **V_FREXP_EXP_I32_F64** - Extract exponent double

### Bit Scanning
- **V_FFBH_U32** - Find first bit high unsigned
- **V_FFBH_I32** - Find first bit high signed
- **V_FFBL_B32** - Find first bit low
- **V_CLREXCP** - Clear exception

### Relative Moves
- **V_MOVRELD_B32** - Move relative destination
- **V_MOVRELS_B32** - Move relative source
- **V_MOVRELSD_B32** - Move relative source/dest
- **V_MOVRELSD_2_B32** - Move relative source/dest 2

### Lane Operations
- **V_READFIRSTLANE_B32** - Copy first active lane to SGPR
- **V_SAT_PK_U8_I16** - Saturate pack unsigned byte

## VECTOR COMPARE INSTRUCTIONS (VOPC)
Generate exec mask based on comparisons.

### Float Comparisons (all types: F16, F32, F64)
- **V_CMP_F_Fx** - Always false
- **V_CMP_LT_Fx** - Less than
- **V_CMP_EQ_Fx** - Equal
- **V_CMP_LE_Fx** - Less or equal
- **V_CMP_GT_Fx** - Greater than
- **V_CMP_LG_Fx** - Not equal
- **V_CMP_GE_Fx** - Greater or equal
- **V_CMP_O_Fx** - Ordered (not NaN)
- **V_CMP_U_Fx** - Unordered (NaN)
- **V_CMP_NGE_Fx** - Not greater or equal
- **V_CMP_NLG_Fx** - Not not-equal
- **V_CMP_NGT_Fx** - Not greater than
- **V_CMP_NLE_Fx** - Not less or equal
- **V_CMP_NEQ_Fx** - Not equal (unordered)
- **V_CMP_NLT_Fx** - Not less than
- **V_CMP_TRU_Fx** - Always true
- **V_CMP_CLASS_Fx** - IEEE class test

### Integer Comparisons (all types: I16, I32, I64, U16, U32, U64)
- **V_CMP_F_Ix** - Always false
- **V_CMP_LT_Ix** - Less than
- **V_CMP_EQ_Ix** - Equal
- **V_CMP_LE_Ix** - Less or equal
- **V_CMP_GT_Ix** - Greater than
- **V_CMP_NE_Ix** - Not equal
- **V_CMP_GE_Ix** - Greater or equal
- **V_CMP_T_Ix** - Always true

### CMPX Variants (update EXEC)
All above comparisons have **V_CMPX_** variants that directly update the EXEC mask instead of writing to a scalar register.

## VECTOR ALU INSTRUCTIONS (VOP3A/VOP3B)
Three-operand vector operations with extended capabilities.

### Integer Arithmetic (Extended)
- **V_ADD_CO_U32** - Add with carry out
- **V_SUB_CO_U32** - Subtract with carry out
- **V_ADDC_CO_U32** - Add with carry in/out
- **V_ADD_NC_I16** - Add no carry INT16
- **V_ADD_NC_I32** - Add no carry INT32
- **V_ADD_NC_U16** - Add no carry UINT16
- **V_ADD3_U32** - Add three operands
- **V_LSHL_ADD_U32** - Shift left and add
- **V_ADD_LSHL_U32** - Add and shift left
- **V_MUL_LO_U16** - Multiply low UINT16
- **V_MUL_LO_U32** - Multiply low UINT32
- **V_MUL_HI_U32** - Multiply high UINT32
- **V_MUL_HI_I32** - Multiply high INT32

### Multiply-Add
- **V_MAD_I16** - Multiply-add INT16
- **V_MAD_U16** - Multiply-add UINT16
- **V_MAD_I32_I16** - Multiply-add INT32 from INT16
- **V_MAD_U32_U16** - Multiply-add UINT32 from UINT16
- **V_MAD_I32_I24** - Multiply-add INT24
- **V_MAD_U32_U24** - Multiply-add UINT24
- **V_MAD_I64_I32** - Multiply-add INT64 from INT32
- **V_MAD_U64_U32** - Multiply-add UINT64 from UINT32

### Float Arithmetic (Extended)
- **V_ADD_F64** - Add double precision
- **V_MUL_F64** - Multiply double precision
- **V_MIN_F64** - Minimum double precision
- **V_MAX_F64** - Maximum double precision
- **V_FMA_F16** - Fused multiply-add half
- **V_FMA_F32** - Fused multiply-add single
- **V_FMA_F64** - Fused multiply-add double
- **V_FMA_LEGACY_F32** - Legacy FMA single

### Min/Max Extended
- **V_MIN_I16** - Minimum INT16
- **V_MIN_U16** - Minimum UINT16
- **V_MAX_I16** - Maximum INT16
- **V_MAX_U16** - Maximum UINT16
- **V_MIN3_I16** - Minimum of three INT16
- **V_MIN3_I32** - Minimum of three INT32
- **V_MIN3_U16** - Minimum of three UINT16
- **V_MIN3_U32** - Minimum of three UINT32
- **V_MIN3_F16** - Minimum of three FP16
- **V_MIN3_F32** - Minimum of three FP32
- **V_MAX3_I16** - Maximum of three INT16
- **V_MAX3_I32** - Maximum of three INT32
- **V_MAX3_U16** - Maximum of three UINT16
- **V_MAX3_U32** - Maximum of three UINT32
- **V_MAX3_F16** - Maximum of three FP16
- **V_MAX3_F32** - Maximum of three FP32

### Median
- **V_MED3_I16** - Median of three INT16
- **V_MED3_I32** - Median of three INT32
- **V_MED3_U16** - Median of three UINT16
- **V_MED3_U32** - Median of three UINT32
- **V_MED3_F16** - Median of three FP16
- **V_MED3_F32** - Median of three FP32

### Bitwise Extended
- **V_LSHLREV_B16** - Shift left reversed INT16
- **V_LSHLREV_B64** - Shift left reversed INT64
- **V_LSHRREV_B16** - Shift right reversed UINT16
- **V_LSHRREV_B64** - Shift right reversed UINT64
- **V_ASHRREV_I16** - Arithmetic shift right INT16
- **V_ASHRREV_I64** - Arithmetic shift right INT64
- **V_AND_OR_B32** - (A & B) | C
- **V_OR3_B32** - A | B | C
- **V_LSHL_OR_B32** - (A << B) | C

### Bit Field Operations
- **V_BFE_U32** - Bit field extract unsigned
- **V_BFE_I32** - Bit field extract signed
- **V_BFI_B32** - Bit field insert
- **V_BFM_B32** - Bit field mask
- **V_BCNT_U32_B32** - Bit count
- **V_ALIGNBIT_B32** - Align bit
- **V_ALIGNBYTE_B32** - Align byte

### Packing Operations
- **V_CVT_PKNORM_I16_F16** - Pack normalized INT16 from FP16
- **V_CVT_PKNORM_I16_F32** - Pack normalized INT16 from FP32
- **V_CVT_PKNORM_U16_F16** - Pack normalized UINT16 from FP16
- **V_CVT_PKNORM_U16_F32** - Pack normalized UINT16 from FP32
- **V_CVT_PK_U16_U32** - Pack UINT16 from UINT32
- **V_CVT_PK_I16_I32** - Pack INT16 from INT32
- **V_CVT_PK_U8_F32** - Pack UINT8 from FP32
- **V_PACK_B32_F16** - Pack FP16 to B32

### Interpolation
- **V_INTERP_P1LL_F16** - Interpolate parameter FP16 P1
- **V_INTERP_P1LV_F16** - Interpolate parameter FP16 P1 variant
- **V_INTERP_P2_F16** - Interpolate parameter FP16 P2

### Division/Scale
- **V_DIV_SCALE_F32** - Division scale FP32
- **V_DIV_SCALE_F64** - Division scale FP64
- **V_DIV_FMAS_F32** - Division FMAS FP32
- **V_DIV_FMAS_F64** - Division FMAS FP64
- **V_DIV_FIXUP_F16** - Division fixup FP16
- **V_DIV_FIXUP_F32** - Division fixup FP32
- **V_DIV_FIXUP_F64** - Division fixup FP64

### Float Operations
- **V_LDEXP_F32** - Load exponent FP32
- **V_LDEXP_F64** - Load exponent FP64
- **V_LERP_U8** - Linear interpolation UINT8
- **V_MULLIT_F32** - Multiply lit

### Cubemap Operations
- **V_CUBEID_F32** - Cubemap face ID
- **V_CUBESC_F32** - Cubemap S coordinate
- **V_CUBETC_F32** - Cubemap T coordinate
- **V_CUBEMA_F32** - Cubemap major axis

### SAD Operations
- **V_MSAD_U8** - Masked sum of absolute differences
- **V_MQSAD_U32_U8** - Masked quad sum of absolute differences
- **V_MQSAD_PK_U16_U8** - Packed masked quad SAD

### Lane Operations
- **V_MBCNT_LO_U32_B32** - Masked bit count low
- **V_MBCNT_HI_U32_B32** - Masked bit count high
- **V_PERMLANE16_B32** - Permute within 16 lanes
- **V_PERMLANEX16_B32** - Permute across 16 lanes

## VOP3P INSTRUCTIONS
Packed operations (dual 16-bit).

- **V_PK_ADD_F16** - Packed add FP16
- **V_PK_MUL_F16** - Packed multiply FP16
- **V_PK_MIN_F16** - Packed minimum FP16
- **V_PK_MAX_F16** - Packed maximum FP16
- **V_PK_FMA_F16** - Packed FMA FP16
- **V_PK_ADD_I16** - Packed add INT16
- **V_PK_SUB_I16** - Packed subtract INT16
- **V_PK_ADD_U16** - Packed add UINT16
- **V_PK_SUB_U16** - Packed subtract UINT16
- **V_PK_MUL_LO_U16** - Packed multiply low UINT16
- **V_PK_MIN_I16** - Packed minimum INT16
- **V_PK_MAX_I16** - Packed maximum INT16
- **V_PK_MIN_U16** - Packed minimum UINT16
- **V_PK_MAX_U16** - Packed maximum UINT16
- **V_PK_LSHLREV_B16** - Packed shift left
- **V_PK_LSHRREV_B16** - Packed shift right
- **V_PK_ASHRREV_I16** - Packed arithmetic shift right

## LDS & GDS INSTRUCTIONS
Local Data Share (LDS) and Global Data Share (GDS) operations.

### Read Operations
- **DS_READ_B32** - Read dword
- **DS_READ_B64** - Read qword
- **DS_READ_B96** - Read 3 dwords
- **DS_READ_B128** - Read 4 dwords
- **DS_READ2_B32** - Read 2 separate dwords
- **DS_READ2_B64** - Read 2 separate qwords
- **DS_READ2ST64_B32** - Read 2 with stride 64
- **DS_READ2ST64_B64** - Read 2 qwords with stride 64
- **DS_READ_U8** - Read unsigned byte
- **DS_READ_I8** - Read signed byte
- **DS_READ_U16** - Read unsigned short
- **DS_READ_I16** - Read signed short
- **DS_READ_U8_D16** - Read byte to D16
- **DS_READ_I8_D16** - Read signed byte to D16
- **DS_READ_U16_D16** - Read short to D16
- **DS_READ_U8_D16_HI** - Read byte to D16 high
- **DS_READ_I8_D16_HI** - Read signed byte to D16 high
- **DS_READ_U16_D16_HI** - Read short to D16 high
- **DS_READ_ADDTID_B32** - Read with thread ID offset

### Write Operations
- **DS_WRITE_B8** - Write byte
- **DS_WRITE_B16** - Write short
- **DS_WRITE_B32** - Write dword
- **DS_WRITE_B64** - Write qword
- **DS_WRITE_B96** - Write 3 dwords
- **DS_WRITE_B128** - Write 4 dwords
- **DS_WRITE2_B32** - Write 2 separate dwords
- **DS_WRITE2_B64** - Write 2 separate qwords
- **DS_WRITE2ST64_B32** - Write 2 with stride 64
- **DS_WRITE2ST64_B64** - Write 2 qwords with stride 64
- **DS_WRITE_B8_D16_HI** - Write byte from D16 high
- **DS_WRITE_B16_D16_HI** - Write short from D16 high
- **DS_WRITE_ADDTID_B32** - Write with thread ID offset

### Atomic Operations
- **DS_ADD_U32** - Atomic add 32-bit
- **DS_ADD_U64** - Atomic add 64-bit
- **DS_SUB_U32** - Atomic subtract 32-bit
- **DS_SUB_U64** - Atomic subtract 64-bit
- **DS_RSUB_U32** - Atomic reverse subtract 32-bit
- **DS_RSUB_U64** - Atomic reverse subtract 64-bit
- **DS_INC_U32** - Atomic increment 32-bit
- **DS_INC_U64** - Atomic increment 64-bit
- **DS_DEC_U32** - Atomic decrement 32-bit
- **DS_DEC_U64** - Atomic decrement 64-bit
- **DS_MIN_I32** - Atomic minimum signed 32-bit
- **DS_MIN_U32** - Atomic minimum unsigned 32-bit
- **DS_MIN_I64** - Atomic minimum signed 64-bit
- **DS_MIN_U64** - Atomic minimum unsigned 64-bit
- **DS_MAX_I32** - Atomic maximum signed 32-bit
- **DS_MAX_U32** - Atomic maximum unsigned 32-bit
- **DS_MAX_I64** - Atomic maximum signed 64-bit
- **DS_MAX_U64** - Atomic maximum unsigned 64-bit
- **DS_AND_B32** - Atomic AND 32-bit
- **DS_AND_B64** - Atomic AND 64-bit
- **DS_OR_B32** - Atomic OR 32-bit
- **DS_OR_B64** - Atomic OR 64-bit
- **DS_XOR_B32** - Atomic XOR 32-bit
- **DS_XOR_B64** - Atomic XOR 64-bit
- **DS_MSKOR_B32** - Atomic masked OR 32-bit
- **DS_MSKOR_B64** - Atomic masked OR 64-bit
- **DS_CMPST_B32** - Atomic compare-swap 32-bit
- **DS_CMPST_B64** - Atomic compare-swap 64-bit
- **DS_ADD_F32** - Atomic add FP32
- **DS_MIN_F32** - Atomic minimum FP32
- **DS_MAX_F32** - Atomic maximum FP32
- **DS_MIN_F64** - Atomic minimum FP64
- **DS_MAX_F64** - Atomic maximum FP64
- **DS_CMPST_F32** - Atomic compare-swap FP32
- **DS_CMPST_F64** - Atomic compare-swap FP64

### Atomic With Return
All above atomics have **_RTN** variants that return the original value.

### Exchange Operations
- **DS_WRXCHG_RTN_B32** - Write-exchange 32-bit
- **DS_WRXCHG_RTN_B64** - Write-exchange 64-bit
- **DS_WRXCHG2_RTN_B32** - Write-exchange 2 dwords
- **DS_WRXCHG2_RTN_B64** - Write-exchange 2 qwords
- **DS_WRXCHG2ST64_RTN_B32** - Write-exchange 2 with stride
- **DS_CONDXCHG32_RTN_B32** - Conditional exchange

### Permute Operations
- **DS_SWIZZLE_B32** - Lane swizzle
- **DS_PERMUTE_B32** - Lane permute
- **DS_BPERMUTE_B32** - Backward permute

### Append/Consume
- **DS_APPEND** - Append to buffer
- **DS_CONSUME** - Consume from buffer
- **DS_ORDERED_COUNT** - Ordered counter

### GDS Semaphore/Barrier
- **DS_GWS_INIT** - Initialize GWS
- **DS_GWS_SEMA_V** - GWS semaphore V
- **DS_GWS_SEMA_P** - GWS semaphore P
- **DS_GWS_SEMA_BR** - GWS semaphore barrier
- **DS_GWS_SEMA_RELEASE** - GWS semaphore release
- **DS_GWS_BARRIER** - GWS barrier

### Miscellaneous
- **DS_NOP** - No operation
- **DS_WRAP_RTN_B32** - Wrap with return

## IMAGE INSTRUCTIONS (MIMG)
Texture sampling and image access.

### Sample Operations
- **IMAGE_SAMPLE** - Basic sample
- **IMAGE_SAMPLE_CL** - Sample with LOD clamp
- **IMAGE_SAMPLE_L** - Sample with explicit LOD
- **IMAGE_SAMPLE_LZ** - Sample LOD zero
- **IMAGE_SAMPLE_B** - Sample with bias
- **IMAGE_SAMPLE_B_CL** - Sample with bias and clamp
- **IMAGE_SAMPLE_D** - Sample with derivatives
- **IMAGE_SAMPLE_D_CL** - Sample with derivatives and clamp
- **IMAGE_SAMPLE_C** - Sample with depth compare
- **IMAGE_SAMPLE_C_CL** - Compare sample with clamp
- **IMAGE_SAMPLE_C_L** - Compare sample with LOD
- **IMAGE_SAMPLE_C_LZ** - Compare sample LOD zero
- **IMAGE_SAMPLE_C_B** - Compare sample with bias
- **IMAGE_SAMPLE_C_B_CL** - Compare sample bias clamp
- **IMAGE_SAMPLE_C_D** - Compare sample with derivatives
- **IMAGE_SAMPLE_C_D_CL** - Compare sample derivatives clamp

### Sample with Offset
All above have **_O** variants for coordinate offset.

### 16-bit Derivatives
Many derivative operations have **_G16** variants using FP16 gradients.

### Gather Operations
- **IMAGE_GATHER4** - Gather 4 texels
- **IMAGE_GATHER4_CL** - Gather with clamp
- **IMAGE_GATHER4_L** - Gather with LOD
- **IMAGE_GATHER4_LZ** - Gather LOD zero
- **IMAGE_GATHER4_B** - Gather with bias
- **IMAGE_GATHER4_B_CL** - Gather bias clamp
- **IMAGE_GATHER4_C** - Gather with compare
- **IMAGE_GATHER4_C_CL** - Gather compare clamp
- **IMAGE_GATHER4_C_L** - Gather compare LOD
- **IMAGE_GATHER4_C_LZ** - Gather compare LOD zero
- **IMAGE_GATHER4_C_B** - Gather compare bias
- **IMAGE_GATHER4_C_B_CL** - Gather compare bias clamp

### Gather with Offset
All gather operations have **_O** variants.

### Load/Store
- **IMAGE_LOAD** - Load from image
- **IMAGE_LOAD_MIP** - Load with mip level
- **IMAGE_LOAD_PCK** - Load packed
- **IMAGE_LOAD_PCK_SGN** - Load packed signed
- **IMAGE_LOAD_MIP_PCK** - Load mip packed
- **IMAGE_LOAD_MIP_PCK_SGN** - Load mip packed signed
- **IMAGE_MSAA_LOAD** - Load MSAA sample
- **IMAGE_STORE** - Store to image
- **IMAGE_STORE_MIP** - Store with mip level
- **IMAGE_STORE_PCK** - Store packed
- **IMAGE_STORE_MIP_PCK** - Store mip packed

### Query Operations
- **IMAGE_GET_RESINFO** - Get resource info
- **IMAGE_GET_LOD** - Get LOD

### Atomic Operations
- **IMAGE_ATOMIC_SWAP** - Atomic swap
- **IMAGE_ATOMIC_CMPSWAP** - Atomic compare-swap
- **IMAGE_ATOMIC_ADD** - Atomic add
- **IMAGE_ATOMIC_SUB** - Atomic subtract
- **IMAGE_ATOMIC_SMIN** - Atomic minimum signed
- **IMAGE_ATOMIC_UMIN** - Atomic minimum unsigned
- **IMAGE_ATOMIC_SMAX** - Atomic maximum signed
- **IMAGE_ATOMIC_UMAX** - Atomic maximum unsigned
- **IMAGE_ATOMIC_AND** - Atomic AND
- **IMAGE_ATOMIC_OR** - Atomic OR
- **IMAGE_ATOMIC_XOR** - Atomic XOR
- **IMAGE_ATOMIC_INC** - Atomic increment
- **IMAGE_ATOMIC_DEC** - Atomic decrement
- **IMAGE_ATOMIC_FCMPSWAP** - Atomic FP compare-swap
- **IMAGE_ATOMIC_FMIN** - Atomic FP minimum
- **IMAGE_ATOMIC_FMAX** - Atomic FP maximum

### Ray Tracing
- **IMAGE_BVH_INTERSECT_RAY** - BVH ray intersection
- **IMAGE_BVH64_INTERSECT_RAY** - 64-bit BVH intersection

## BUFFER INSTRUCTIONS
Buffer memory access (structured).

### Load Operations
- **BUFFER_LOAD_UBYTE** - Load unsigned byte
- **BUFFER_LOAD_SBYTE** - Load signed byte
- **BUFFER_LOAD_USHORT** - Load unsigned short
- **BUFFER_LOAD_SSHORT** - Load signed short
- **BUFFER_LOAD_DWORD** - Load dword
- **BUFFER_LOAD_DWORDX2** - Load 2 dwords
- **BUFFER_LOAD_DWORDX3** - Load 3 dwords
- **BUFFER_LOAD_DWORDX4** - Load 4 dwords
- **BUFFER_LOAD_UBYTE_D16** - Load byte to D16
- **BUFFER_LOAD_UBYTE_D16_HI** - Load byte to D16 high
- **BUFFER_LOAD_SBYTE_D16** - Load signed byte to D16
- **BUFFER_LOAD_SBYTE_D16_HI** - Load signed byte to D16 high
- **BUFFER_LOAD_SHORT_D16** - Load short to D16
- **BUFFER_LOAD_SHORT_D16_HI** - Load short to D16 high

### Format Loads
- **BUFFER_LOAD_FORMAT_X** - Load 1 component with format
- **BUFFER_LOAD_FORMAT_XY** - Load 2 components
- **BUFFER_LOAD_FORMAT_XYZ** - Load 3 components
- **BUFFER_LOAD_FORMAT_XYZW** - Load 4 components
- **BUFFER_LOAD_FORMAT_D16_X** - Format load to D16
- **BUFFER_LOAD_FORMAT_D16_XY** - Format load 2 to D16
- **BUFFER_LOAD_FORMAT_D16_XYZ** - Format load 3 to D16
- **BUFFER_LOAD_FORMAT_D16_XYZW** - Format load 4 to D16
- **BUFFER_LOAD_FORMAT_D16_HI_X** - Format load to D16 high

### Store Operations
- **BUFFER_STORE_BYTE** - Store byte
- **BUFFER_STORE_BYTE_D16_HI** - Store byte from D16 high
- **BUFFER_STORE_SHORT** - Store short
- **BUFFER_STORE_SHORT_D16_HI** - Store short from D16 high
- **BUFFER_STORE_DWORD** - Store dword
- **BUFFER_STORE_DWORDX2** - Store 2 dwords
- **BUFFER_STORE_DWORDX3** - Store 3 dwords
- **BUFFER_STORE_DWORDX4** - Store 4 dwords

### Format Stores
- **BUFFER_STORE_FORMAT_X** - Store 1 component with format
- **BUFFER_STORE_FORMAT_XY** - Store 2 components
- **BUFFER_STORE_FORMAT_XYZ** - Store 3 components
- **BUFFER_STORE_FORMAT_XYZW** - Store 4 components
- **BUFFER_STORE_FORMAT_D16_X** - Format store from D16
- **BUFFER_STORE_FORMAT_D16_XY** - Format store 2 from D16
- **BUFFER_STORE_FORMAT_D16_XYZ** - Format store 3 from D16
- **BUFFER_STORE_FORMAT_D16_XYZW** - Format store 4 from D16
- **BUFFER_STORE_FORMAT_D16_HI_X** - Format store from D16 high

### Atomic Operations
- **BUFFER_ATOMIC_SWAP** - Atomic swap
- **BUFFER_ATOMIC_CMPSWAP** - Atomic compare-swap
- **BUFFER_ATOMIC_ADD** - Atomic add
- **BUFFER_ATOMIC_SUB** - Atomic subtract
- **BUFFER_ATOMIC_SMIN** - Atomic minimum signed
- **BUFFER_ATOMIC_UMIN** - Atomic minimum unsigned
- **BUFFER_ATOMIC_SMAX** - Atomic maximum signed
- **BUFFER_ATOMIC_UMAX** - Atomic maximum unsigned
- **BUFFER_ATOMIC_AND** - Atomic AND
- **BUFFER_ATOMIC_OR** - Atomic OR
- **BUFFER_ATOMIC_XOR** - Atomic XOR
- **BUFFER_ATOMIC_INC** - Atomic increment
- **BUFFER_ATOMIC_DEC** - Atomic decrement
- **BUFFER_ATOMIC_FCMPSWAP** - Atomic FP compare-swap
- **BUFFER_ATOMIC_FMIN** - Atomic FP minimum
- **BUFFER_ATOMIC_FMAX** - Atomic FP maximum
- **BUFFER_ATOMIC_CSUB** - Atomic compare-subtract

### 64-bit Atomics
All above atomics have **_X2** variants for 64-bit operations.

### Cache Operations
- **BUFFER_GL0_INV** - Invalidate GL0 cache
- **BUFFER_GL1_INV** - Invalidate GL1 cache

## FLAT/GLOBAL/SCRATCH INSTRUCTIONS
Flat address space memory access.

### FLAT Load/Store
- **FLAT_LOAD_UBYTE** - Load unsigned byte
- **FLAT_LOAD_SBYTE** - Load signed byte
- **FLAT_LOAD_USHORT** - Load unsigned short
- **FLAT_LOAD_SSHORT** - Load signed short
- **FLAT_LOAD_DWORD** - Load dword
- **FLAT_LOAD_DWORDX2** - Load 2 dwords
- **FLAT_LOAD_DWORDX3** - Load 3 dwords
- **FLAT_LOAD_DWORDX4** - Load 4 dwords
- **FLAT_STORE_BYTE** - Store byte
- **FLAT_STORE_SHORT** - Store short
- **FLAT_STORE_DWORD** - Store dword
- **FLAT_STORE_DWORDX2** - Store 2 dwords
- **FLAT_STORE_DWORDX3** - Store 3 dwords
- **FLAT_STORE_DWORDX4** - Store 4 dwords

### FLAT D16 Operations
- **FLAT_LOAD_UBYTE_D16** - Load byte to D16
- **FLAT_LOAD_UBYTE_D16_HI** - Load byte to D16 high
- **FLAT_LOAD_SBYTE_D16** - Load signed byte to D16
- **FLAT_LOAD_SBYTE_D16_HI** - Load signed byte to D16 high
- **FLAT_LOAD_SHORT_D16** - Load short to D16
- **FLAT_LOAD_SHORT_D16_HI** - Load short to D16 high
- **FLAT_STORE_BYTE_D16_HI** - Store byte from D16 high
- **FLAT_STORE_SHORT_D16_HI** - Store short from D16 high

### FLAT Atomics
- **FLAT_ATOMIC_SWAP** - Atomic swap
- **FLAT_ATOMIC_CMPSWAP** - Atomic compare-swap
- **FLAT_ATOMIC_ADD** - Atomic add
- **FLAT_ATOMIC_SUB** - Atomic subtract
- **FLAT_ATOMIC_SMIN** - Atomic minimum signed
- **FLAT_ATOMIC_UMIN** - Atomic minimum unsigned
- **FLAT_ATOMIC_SMAX** - Atomic maximum signed
- **FLAT_ATOMIC_UMAX** - Atomic maximum unsigned
- **FLAT_ATOMIC_AND** - Atomic AND
- **FLAT_ATOMIC_OR** - Atomic OR
- **FLAT_ATOMIC_XOR** - Atomic XOR
- **FLAT_ATOMIC_INC** - Atomic increment
- **FLAT_ATOMIC_DEC** - Atomic decrement
- **FLAT_ATOMIC_FCMPSWAP** - Atomic FP compare-swap
- **FLAT_ATOMIC_FMIN** - Atomic FP minimum
- **FLAT_ATOMIC_FMAX** - Atomic FP maximum

### FLAT 64-bit Atomics
All above have **_X2** variants.

### GLOBAL Load/Store
- **GLOBAL_LOAD_UBYTE** - Load unsigned byte
- **GLOBAL_LOAD_SBYTE** - Load signed byte
- **GLOBAL_LOAD_USHORT** - Load unsigned short
- **GLOBAL_LOAD_SSHORT** - Load signed short
- **GLOBAL_LOAD_DWORD** - Load dword
- **GLOBAL_LOAD_DWORDX2** - Load 2 dwords
- **GLOBAL_LOAD_DWORDX3** - Load 3 dwords
- **GLOBAL_LOAD_DWORDX4** - Load 4 dwords
- **GLOBAL_LOAD_DWORD_ADDTID** - Load with thread ID offset
- **GLOBAL_STORE_BYTE** - Store byte
- **GLOBAL_STORE_SHORT** - Store short
- **GLOBAL_STORE_DWORD** - Store dword
- **GLOBAL_STORE_DWORDX2** - Store 2 dwords
- **GLOBAL_STORE_DWORDX3** - Store 3 dwords
- **GLOBAL_STORE_DWORDX4** - Store 4 dwords
- **GLOBAL_STORE_DWORD_ADDTID** - Store with thread ID offset

### GLOBAL D16 Operations
- **GLOBAL_LOAD_UBYTE_D16** - Load byte to D16
- **GLOBAL_LOAD_UBYTE_D16_HI** - Load byte to D16 high
- **GLOBAL_LOAD_SBYTE_D16** - Load signed byte to D16
- **GLOBAL_LOAD_SBYTE_D16_HI** - Load signed byte to D16 high
- **GLOBAL_LOAD_SHORT_D16** - Load short to D16
- **GLOBAL_LOAD_SHORT_D16_HI** - Load short to D16 high
- **GLOBAL_STORE_BYTE_D16_HI** - Store byte from D16 high
- **GLOBAL_STORE_SHORT_D16_HI** - Store short from D16 high

### GLOBAL Atomics
- **GLOBAL_ATOMIC_SWAP** - Atomic swap
- **GLOBAL_ATOMIC_CMPSWAP** - Atomic compare-swap
- **GLOBAL_ATOMIC_ADD** - Atomic add
- **GLOBAL_ATOMIC_SUB** - Atomic subtract
- **GLOBAL_ATOMIC_CSUB** - Atomic compare-subtract
- **GLOBAL_ATOMIC_SMIN** - Atomic minimum signed
- **GLOBAL_ATOMIC_UMIN** - Atomic minimum unsigned
- **GLOBAL_ATOMIC_SMAX** - Atomic maximum signed
- **GLOBAL_ATOMIC_UMAX** - Atomic maximum unsigned
- **GLOBAL_ATOMIC_AND** - Atomic AND
- **GLOBAL_ATOMIC_OR** - Atomic OR
- **GLOBAL_ATOMIC_XOR** - Atomic XOR
- **GLOBAL_ATOMIC_INC** - Atomic increment
- **GLOBAL_ATOMIC_DEC** - Atomic decrement
- **GLOBAL_ATOMIC_FCMPSWAP** - Atomic FP compare-swap
- **GLOBAL_ATOMIC_FMIN** - Atomic FP minimum
- **GLOBAL_ATOMIC_FMAX** - Atomic FP maximum

### GLOBAL 64-bit Atomics
All above have **_X2** variants.

### SCRATCH Operations
- **SCRATCH_LOAD_UBYTE** - Load unsigned byte
- **SCRATCH_LOAD_SBYTE** - Load signed byte
- **SCRATCH_LOAD_USHORT** - Load unsigned short
- **SCRATCH_LOAD_SSHORT** - Load signed short
- **SCRATCH_LOAD_DWORD** - Load dword
- **SCRATCH_LOAD_DWORDX2** - Load 2 dwords
- **SCRATCH_LOAD_DWORDX3** - Load 3 dwords
- **SCRATCH_LOAD_DWORDX4** - Load 4 dwords
- **SCRATCH_STORE_BYTE** - Store byte
- **SCRATCH_STORE_SHORT** - Store short
- **SCRATCH_STORE_DWORD** - Store dword
- **SCRATCH_STORE_DWORDX2** - Store 2 dwords
- **SCRATCH_STORE_DWORDX3** - Store 3 dwords
- **SCRATCH_STORE_DWORDX4** - Store 4 dwords

### SCRATCH D16 Operations
- **SCRATCH_LOAD_UBYTE_D16** - Load byte to D16
- **SCRATCH_LOAD_UBYTE_D16_HI** - Load byte to D16 high
- **SCRATCH_LOAD_SBYTE_D16** - Load signed byte to D16
- **SCRATCH_LOAD_SBYTE_D16_HI** - Load signed byte to D16 high
- **SCRATCH_LOAD_SHORT_D16** - Load short to D16
- **SCRATCH_LOAD_SHORT_D16_HI** - Load short to D16 high
- **SCRATCH_STORE_BYTE_D16_HI** - Store byte from D16 high
- **SCRATCH_STORE_SHORT_D16_HI** - Store short from D16 high

## SCALAR MEMORY INSTRUCTIONS (SMEM)
Scalar memory reads through scalar data cache.

- **S_LOAD_DWORD** - Load scalar dword
- **S_LOAD_DWORDX2** - Load 2 scalar dwords
- **S_LOAD_DWORDX4** - Load 4 scalar dwords
- **S_LOAD_DWORDX8** - Load 8 scalar dwords
- **S_LOAD_DWORDX16** - Load 16 scalar dwords
- **S_BUFFER_LOAD_DWORD** - Buffer load scalar dword
- **S_BUFFER_LOAD_DWORDX2** - Buffer load 2 scalar dwords
- **S_BUFFER_LOAD_DWORDX4** - Buffer load 4 scalar dwords
- **S_BUFFER_LOAD_DWORDX8** - Buffer load 8 scalar dwords
- **S_BUFFER_LOAD_DWORDX16** - Buffer load 16 scalar dwords
- **S_DCACHE_INV** - Invalidate data cache
- **S_DCACHE_WB** - Write-back data cache
- **S_DCACHE_INV_VOL** - Invalidate volatile
- **S_DCACHE_WB_INV** - Write-back and invalidate
- **S_MEMTIME** - Get memory timestamp
- **S_MEMREALTIME** - Get real-time memory timestamp
- **S_ATC_PROBE** - ATC probe
- **S_ATC_PROBE_BUFFER** - ATC probe buffer
- **S_GL1_INV** - Invalidate GL1 cache

## BRANCH AND MESSAGE INSTRUCTIONS

### Conditional Branch
- **S_CBRANCH_SCC0** - Branch if SCC == 0
- **S_CBRANCH_SCC1** - Branch if SCC == 1
- **S_CBRANCH_VCCZ** - Branch if VCC all zero
- **S_CBRANCH_VCCNZ** - Branch if VCC not all zero
- **S_CBRANCH_EXECZ** - Branch if EXEC all zero
- **S_CBRANCH_EXECNZ** - Branch if EXEC not all zero
- **S_CBRANCH_CDBGSYS** - Conditional branch debug system
- **S_CBRANCH_CDBGUSER** - Conditional branch debug user
- **S_CBRANCH_CDBGSYS_OR_USER** - Branch debug sys or user
- **S_CBRANCH_CDBGSYS_AND_USER** - Branch debug sys and user

### Unconditional Branch
- **S_BRANCH** - Unconditional branch
- **S_SETPRIO** - Set priority
- **S_SLEEP** - Sleep
- **S_SETPC_B64** - Set program counter
- **S_SWAPPC_B64** - Swap program counter
- **S_BARRIER** - Barrier
- **S_TTRACEDATA** - Thread trace data
- **S_ICACHE_INV** - Invalidate instruction cache
- **S_ENDPGM** - End program
- **S_ENDPGM_SAVED** - End program saved
- **S_ENDPGM_ORDERED_PS_DONE** - End program ordered PS
- **S_WAKEUP** - Wakeup
- **S_SENDMSG** - Send message
- **S_SENDMSGHALT** - Send message and halt
- **S_TRAP** - Trap
- **S_CODE_END** - Code end marker
- **S_INST_PREFETCH** - Instruction prefetch
- **S_CLAUSE** - Start instruction clause
- **S_WAITCNT_DEPCTR** - Wait for dependency counter
- **S_ROUND_MODE** - Set rounding mode
- **S_DENORM_MODE** - Set denorm mode
- **S_TTRACEDATA_IMM** - Thread trace immediate data

## EXPORT INSTRUCTIONS

- **EXP** - Export (position, parameter, render target)
  - Targets: MRTZ (depth), NULL, POS0-3, PARAM0-31

## Summary Statistics

- **Scalar Integer/Bitwise Ops**: ~80 instructions
- **Vector Integer Ops**: ~40 base + variants
- **Vector Float Ops (FP16/FP32/FP64)**: ~60 instructions
- **Vector Transcendentals**: ~25 instructions
- **Conversions**: ~35 instructions
- **Compare Operations**: ~150 variants (all type combos)
- **LDS/GDS Operations**: ~120 instructions
- **Image Operations**: ~100 instructions
- **Buffer Operations**: ~90 instructions
- **Flat/Global/Scratch**: ~150 instructions
- **Atomic Operations**: ~80 variants across all memory types

**Total Unique Operations**: 900+ instruction variants

## Key Architectural Features

- **Wave Size**: 32 threads per wave (RDNA2)
- **VGPRs**: Up to 256 vector registers per wave
- **SGPRs**: Up to 106 scalar registers per wave
- **LDS**: 128 KB per compute unit (64 banks × 512 entries × 4 bytes)
- **GDS**: 64 KB global (32 banks × 512 entries × 4 bytes)
- **Execution Mask**: EXEC register controls active lanes
- **IEEE 754 Compliance**: Full FP16/FP32/FP64 support
- **Native FP16**: Packed dual FP16 operations via VOP3P
- **Dot Product Acceleration**: Hardware INT8/FP16 dot products
- **Ray Tracing**: BVH intersection acceleration

## Notes

- Most instructions have multiple encoding formats (VOP2, VOP3, etc.)
- Many operations support input modifiers (abs, neg, clamp)
- Atomics available across LDS, GDS, buffer, image, and flat address spaces
- D16 operations pack 16-bit data into 32-bit registers
- Memory operations support various data types and widths
- Extensive support for graphics-specific operations (interpolation, sampling)