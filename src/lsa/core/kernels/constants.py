"""The values the kernels have to agree on, and each kernel's tiling and tuning.

A group is imported by name, as `from lsa.core.kernels.constants import Index`.
The first groups hold what two or more kernels share, the Triton kernels of a step and its CuTeDSL kernel included, so the two read one definition where their bits have to match.
The rest hold one kernel's tiling and tuning, a group a kernel.

Plain classes rather than `enum.Enum`, because these go straight into kernel launches and compile-time shapes where `.value` would be noise.
A CuTeDSL kernel reads `Group.NAME` in its body, since it resolves a plain int off a class at trace time.
A Triton `@triton.jit` body reads only a global that is a `tl.constexpr`, so each Triton file binds what its bodies need once at module level, as `NAME: tl.constexpr = tl.constexpr(Group.NAME)`.
"""

from typing import ClassVar


class Warp:
    """What a warp is made of."""

    # Lanes of a warp
    SIZE: int = 32


class Index:
    """The index space the kernels score in."""

    # Index dims the kernels take - a row of 128 bytes, the widest swizzle
    DIM: int = 64


class PackedKey:
    """The packed key a score and its bucket travel in, as Triton and CuTeDSL both build it."""

    # The empty key, which sorts last - the least int64, the key of no bucket and the key past a row's listed count
    EMPTY: int = -(2**63)
    # The greatest key
    MAX: int = 0x7FFFFFFFFFFFFFFF
    # The cost key's low bits name the column, as the Triton key does
    INDEX_BITS: int = 12


class Radix:
    """The radix pass of the beam step and of the expansion by threshold."""

    # Bins of a radix pass, and the bins each lane of the warp that finds a pass's digit holds
    BINS: int = 256
    BINS_PER_LANE: int = BINS // Warp.SIZE


class ThresholdEstimate:
    """The threshold estimate, which CuTeDSL has to match Triton's bit for bit."""

    # Samples a Triton program of the histogram adds as one block - the columns an exact pick's program reads at a time
    BLOCK: int = 4096
    BINS: int = 64
    # Triton's slack under a bin's edge, so a sampled bucket scored at it again still counts
    # Below a bin edge the threshold sits, as a share of the query's norm - more than a sample's tensor-core score
    # and the per-query score differ by
    EDGE_SLACK: float = 2**-14


class ShapedRead:
    """The shaped read, which scores a bucket against its planes."""

    # Bits of the query levels a packed basis is scored against
    QUERY_BITS: int = 8
    # Key positions a shape build reads at a time - a chunk of buckets' members gathered at once
    KEYS: int = 1 << 17
    # A plane gap's widening, past the fp32 rounding of the planes' levels and of the gap itself
    GAP_SLACK: float = 1 + 2**-8
    # Sign directions a shaped level's kernel takes, and the basis words of one
    RANK: int = 2
    WORDS: int = Index.DIM // Warp.SIZE
    # Index dims a program of the shaped read takes at a time, and the columns of its score rows, its planes' spread and
    # a paged scan's rows
    DIM_CHUNK: int = 128
    SCORE_BLOCK: int = 32
    SPREAD_BLOCK: int = 64
    PAGED_ROWS: int = 64


class DenseScoring:
    """The dense scoring of a row, which CuTeDSL spans as Triton does."""

    # Buckets a CTA of the dense scoring takes, as the Triton dense scoring spans them
    # Buckets one program of a dense row scores, and rows scored densely at once
    SPAN: int = 8192
    ROWS: int = 256


class ScoreGroups:
    """How the candidate scoring and the decode scan split a CTA into groups of lanes."""

    # The lanes and dims of each that add one score, as Triton lays a 64-dim dot out
    GROUP_LANES: int = 8
    DIMS_PER_LANE: int = Index.DIM // GROUP_LANES
    THREADS: int = 128
    GROUPS: int = THREADS // GROUP_LANES
    # Listed buckets a group of lanes has in flight
    IN_FLIGHT: int = 8


class SharedBucketScan:
    """The bucket scan, the same on either GPU where a caller sees it."""

    # Tiles a CTA scans before its rows reserve their place in the global listing, at most and at least - within
    # them a span is sized so the CTAs fill the GPU's slots about WAVES times over
    SPAN_TILES: tuple[int, ...] = (4, 64)
    # Survivors a row holds in shared memory a span before they move to the global listing - some 11 arrive at 10M,
    # so few rows ever append past it
    ROW_SLOTS: int = 32
    # Tiles a span holds at most, which a queued word's tag in an int16 bounds - the Hopper listing's bound, so a
    # caller's span suits either GPU
    MAX_SPAN_TILES: int = 2**14


class Arch:
    """The GPUs a kernel is written for, as compute capabilities."""

    HOPPER: tuple[int, int] = (9, 0)
    B200: tuple[int, int] = (10, 0)
    B300: tuple[int, int] = (10, 3)
    # The GPUs that have `tcgen05` MMA and tensor memory, which a kernel under cutedsl/blackwell needs
    BLACKWELL: frozenset[tuple[int, int]] = frozenset({B200, B300})


class TritonBeam:
    """The Triton beam step."""

    # Node centroids stored in fp8 hold this times the unit row they stand for, to keep small components in range
    NODE_CENTROID_SCALE: float = 16.0
    # Columns a program of a beam step's scoring and of its level takes at a time, of a candidate listing, and of a
    # listing's length count
    BLOCK: int = 64
    CANDIDATE_BLOCK: int = 1024
    LENGTH_BLOCK: int = 1024


class TritonBlockScores:
    """The Triton block scores."""

    # Columns a program dots against - a leaf block's keys, or the summaries of a group of upper blocks
    COLUMNS: int = 64

    # Key dims accumulated a step
    BLOCK_DIM: int = 64

    # Rows a call scores at most - the launch's second dim
    MAX_ROWS: int = 65535

    # Fewest heads `tl.dot` takes - fewer are padded with zero queries, whose LogSumExp is dropped
    MIN_HEADS: int = 16


class TritonDescend:
    """The Triton keep of a descent."""

    # A kept slot past the row's blocks, which sorts last at the leaves
    NO_BLOCK: int = 1 << 40


class TritonPyramid:
    """The Triton pooling of a pyramid's levels."""

    # Key dims a program pools
    BLOCK_DIM: int = 64

    # Summaries of a level pooled at once
    PARENTS: int = 32


class TritonScanSelect:
    """The Triton fused scan and select."""

    # Visible buckets from which flat's prefill read takes the fused select - under it a dense product and a top-k are
    # cheaper
    SCAN_SELECT_BUCKETS: int = 32768
    # Columns of a beam step's score row from which the level takes the fused select - under it a top-k over the
    # materialized scores is cheaper
    SCAN_SELECT_BEAM_WIDTH: int = 196608
    # Survivors a row aims for and holds, in top-p
    TARGET: float = 1.5
    CAPACITY: int = 3
    # Sampled buckets a row's threshold rests on - its relative spread is about one over the root
    SAMPLE_RANK: int = 64
    # Below the threshold the scan still keeps, as a share of the query's norm - over unit bf16 bucket_centroids of 64
    # the tensor-core and the per-query fp32 dot differ by under 1e-5 of it
    MARGIN: float = 1e-4
    # A shaped score the scan widens by this share of its spread, past the fp32 rounding of the planes' variance
    SPREAD_SLACK: float = 1 + 2**-10
    # Buckets a row's scoring reads at a time - one layout for listed and dense rows keeps their scores equal
    SCORE_BLOCK: int = 128
    # Widest a row lists again, in top-p, before it is scored in full, and entries one wide listing holds
    WIDE: int = 64
    WIDE_ENTRIES: int = 2**26
    # Entries a program of a survivor listing expands at a time, and index dims it takes at a time
    EXPAND: int = 128
    DIM_CHUNK: int = 128


class TritonPaged:
    """The Triton paged scan."""

    # Rows a program scores, and index dims it takes at a time
    ROWS: int = 32
    DIM_CHUNK: int = 128


class TritonExpansion:
    """The Triton expansion by threshold."""

    # Picks a program of the member listing writes at a time
    TILE: int = 128


class TritonTopK:
    """The Triton exact top-k."""

    # Widest a row is listed whole, in k - a sample cannot rank so large a share of a row
    WHOLE: int = 4
    # Most columns a row samples
    SAMPLE_MOST: int = 24576
    # Samples a tight listing's estimate must rank - spread is 1 / sqrt(rank)
    # => a listing misses when it falls a third under its target or past its capacity
    TIGHT_RANK: int = 192
    # Listing (columns expected in k, columns held in k, 1 / share of the rank a bin holds before a split)
    TIGHT: tuple[float, int, int] = (1.4, 2, 5)
    LOOSE: tuple[float, int, int] = (2.0, 4, 3)
    SAMPLE_RANK: int = 256
    LIST_BLOCK: int = 2048
    SEARCH_BLOCK: int = 4096
    # Bits of a column id - a row is at most this wide
    ID_BITS: int = 24
    # Widest a row an exact pick selects where it lies, in columns
    NARROW: int = 4096
    # Samples an exact pick's wide row can be estimated from, and the listing sizes tried for them, in k
    SAMPLES: tuple[int, ...] = (2048, 4096, 8192, 16384)
    TARGETS: tuple[float, ...] = (
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.4,
        1.45,
        1.5,
        1.6,
        1.7,
        1.8,
        2.0,
        2.25,
        2.5,
        3.0,
    )
    # Neighbours a sample takes along - a sector costs the same to read as one score
    RUN: int = 8
    # Standard deviations of the listing's size the estimate must clear its k and its buffer by
    SIGMAS: float = 4.0
    # Least columns a listing holds when no sample plans it tightly
    LOOSE_CAPACITY: int = 8192
    # Columns an exact pick's program reads at a time
    BLOCK: int = ThresholdEstimate.BLOCK
    # Rows under which a row is split across programs, each reading SPLIT_LEAST columns at least
    SPLIT_BELOW: int = 64
    SPLIT_LEAST: int = 8192
    SPLIT_MOST: int = 64
    SPLIT_PROGRAMS: int = 512
    WARPS: int = 8
    SELECT_WARPS: int = 4
    # Columns, warps and loads ahead of a row listed in one program
    # => small programs keep more loads in flight
    ROW_BLOCK: int = 1024
    ROW_WARPS: int = 2
    ROW_STAGES: int = 3
    # Warps an exact pick's estimate runs with
    ESTIMATE_WARPS: int = 4
    # Bits of the score word a whole-row search settles a pass, and the columns it reads at a time
    DIGIT_BITS: int = 4
    DIGIT_BLOCK: int = 512


class TritonWard:
    """The Triton ward clustering."""

    # The Gram matrix's dot - exact, three tf32 passes that match it, or one
    PRECISION: ClassVar[dict[str, int]] = {"ieee": 0, "tf32x3": 1, "tf32": 2}


class CuteBeam:
    """The CuTeDSL beam step."""

    # Threads of a beam step that scores children, and of every other launch - the more loads in flight, the better, until
    # a scoring level's gathers crowd each other out. A scoring level whose shared memory passes SCORING_SHARED_BYTES
    # falls from four CTAs an SM to three and takes the larger CTA too: over 8,192 children and 3,024 open node buckets it
    # ran 1.77 ms at 256 threads and 1.56 at 512, and 1.52 and 1.55 over 2,524. The pick gains as much and never loses
    SCORING_THREADS: int = 256
    READING_THREADS: int = 512
    SCORING_SHARED_BYTES: int = 49 * 1024
    # A row already scored is only read, and twice the threads halve each one's run of it
    # => 1,024 threads
    SCORED_THREADS: int = 1024
    MOST_WARPS: int = SCORED_THREADS // Warp.SIZE
    # The ordered word of -inf - a kept column at it keeps no children
    NEG_INF_WORD: int = 0x007FFFFF
    # Words of shared memory before the row - the histogram, each warp's two counts and the search's state, to 16 bytes
    STATE_WORDS: int = 8
    FIXED_WORDS: int = (Radix.BINS + 2 * MOST_WARPS + STATE_WORDS + 3) // 4 * 4
    # Shared memory a CTA may take on Hopper - the fixed words, the row and the kept columns
    SHARED_BYTES: int = 227 * 1024
    SHARED_WORDS: int = SHARED_BYTES // 4 - FIXED_WORDS
    # Scores a thread loads before it stores any
    COPY_IN_FLIGHT: int = 16
    # Words a row samples, at a fixed stride where they lie, the listing it expects above the sample's threshold and the
    # most it holds, in the keep - a row narrower than SAMPLED_FROM times its keep, or than the sample, is searched whole.
    # Every word of room past the row costs occupancy: a pick of 1,024 over 9,232 columns ran 2.05 ms with room for twice
    # its keep and 1.72 with room for 1.5 times
    SAMPLES: int = 4096
    TARGET: float = 1.15
    HOLD: float = 1.5
    SAMPLED_FROM: int = 8
    # Children a thread gathers before it adds any - enough loads in flight to cover a gather's latency
    IN_FLIGHT: int = 8
    # Bytes a thread of a child holds, as Triton's layout of the child scores holds them
    THREAD_BYTES: int = 16


class CuteDecode:
    """The CuTeDSL decode scan and its select."""

    # Buckets a CTA of the scan takes
    SPAN: int = 256
    # Spans a request's listing is ordered from at most, their places held in shared memory - contexts to 16M
    MOST_SPANS: int = 8192
    SPAN_BITS: int = MOST_SPANS.bit_length() - 1
    # Threads of the threshold and of the ordering, and the bits of a word each pass of their search settles
    SELECT_THREADS: int = 1024
    SELECT_WARPS: int = SELECT_THREADS // Warp.SIZE
    # CTAs a request's ordering is split over, each taking every ORDER_PARTS-th run of a CTA's width of its keys
    ORDER_PARTS: int = 4
    DIGIT_BITS: int = 8
    BINS: int = 1 << DIGIT_BITS
    # Samples of a request a thread of the threshold holds, at most
    SAMPLES_PER_THREAD: int = -(-TritonTopK.SAMPLES[-1] // SELECT_THREADS)
    # The threshold's search settles the top two digits of ten bits, a bin a thread - more than the word's kept bits
    SEARCH_BITS: int = 10
    SEARCH_BINS: int = 1 << SEARCH_BITS
    SEARCH_PASSES: int = 2
    # A histogram of the search holds its bins padded by one in every 32, so a lane's 32 bins sit in distinct banks
    HISTOGRAM_SIZE: int = SEARCH_BINS + SEARCH_BINS // Warp.SIZE
    # Low bits of the sampled threshold's word left clear - a granule of the scores far below the listing's slack
    THRESHOLD_LOW: int = 13


class CuteExpansion:
    """The CuTeDSL expansion by threshold and in rank order."""

    # Threads of the CTA a row takes: 1,024 read a listing faster while rows leave SMs idle, 512 once rows fill the GPU
    # => 1,024 for few rows, 512 for many
    FEW_ROWS_THREADS: int = 1024
    MANY_ROWS_THREADS: int = 512
    # Rows an SM holds before a launch fills the GPU
    ROWS_PER_SM: int = 1
    # The bit that orders a signed key as an unsigned one
    SIGN: int = 2**63
    # A bucket above the last key counts once in the walk and its members in the read, packed in one word
    WALKED: int = 1 << 16
    # Rounds of buckets a warp copies the members of with all their loads in flight
    WALK_IN_FLIGHT: int = 4
    # The search's state words
    PREFIX_HIGH: int = 0
    PREFIX_LOW: int = 1
    ABOVE_MEMBERS: int = 2
    ABOVE_BUCKETS: int = 3
    NONE_REACHES: int = 4
    TIED: int = 5
    CUT_ROW: int = 6
    TAKEN: int = 7
    STATE_WORDS: int = 8


class CuteKeep:
    """The CuTeDSL start-level keep."""

    # Keys a row's listing holds at most, their words in shared memory
    MOST_LISTED: int = 8192
    KEEP_THREADS: int = 256
    KEEP_WARPS: int = KEEP_THREADS // Warp.SIZE


class CuteSpread:
    """The CuTeDSL top-p pick spread over a cluster."""

    # A cluster larger than 8 needs an opt-in and ran slower
    # => 8 CTAs
    CLUSTER: int = 8
    # More threads or fewer ran slower, and below this many columns the single CTA is as fast
    # => 512 threads, spread from 4,096 columns
    SPREAD_THREADS: int = 512
    SPREAD_FROM: int = 4096
    # Past this many rows the clusters of 8 queue for SMs
    # => clusters of 4
    FEW_ROWS: int = 8
    # A word is 4 bytes
    # => 4 passes of a byte each
    PASSES: int = 4
    # Two passes leave a few words under the digits found on a row of scores without heavy ties
    # => at most this many are listed and ranked, one barrier in place of two passes
    GATHER_MAX: int = 256
    # The slice follows a histogram a pass, their sum, each warp's two counts, the state and the words listed and gathered
    # => these fixed words, to 16 bytes
    STATE_WORDS: int = 16
    FIXED_WORDS: int = (
        (PASSES * Radix.BINS + Radix.BINS + 2 * CuteBeam.MOST_WARPS + STATE_WORDS + 2 * GATHER_MAX + 3)
        // 4
        * 4
    )
    # The state's words - the prefix and the rank left at it before each pass and after the last, then the slice's columns
    # above the level and at it, the words under the digits found by the second pass and this CTA's share of them
    # => four words from index 2 * PASSES + 2
    SLICE_ABOVE: int = 2 * PASSES + 2
    SLICE_TIED: int = 2 * PASSES + 3
    BIN_COUNT: int = 2 * PASSES + 4
    LISTED: int = 2 * PASSES + 5


class CuteThresholds:
    """The CuTeDSL threshold estimate."""

    # Rows a CTA holds, samples a tile scores, and the CTA's threads - a warp's 16 rows a thread's two
    ROWS: int = 128
    COLUMNS: int = 64
    THREADS: int = 256
    WARPS: int = THREADS // Warp.SIZE
    SWIZZLE_DIM: int = 64
    # Slots a thread tests together against its row's edge
    HIT_SLOTS: int = 4
    # CTAs an SM holds at once, which caps a thread's registers - the sums pass's, the histogram pass's
    MIN_BLOCKS: ClassVar[dict[bool, int]] = {False: 2, True: 3}
    # The tiles of a Triton program's block
    TILES: int = ThresholdEstimate.BLOCK // COLUMNS
    # Disjoint groups of a row's samples, whose maxima's rank-th greatest bounds its rank-th best - each block of an
    # odd or even place in a set of groups of its own
    GROUPS: int = 128
    GROUP_SETS: int = 2
    # Steps an estimate of a bin's edge moves, in ulps, to meet it
    EDGE_STEPS: int = 8


class CuteTopK:
    """The CuTeDSL top-p pick."""

    # Samples a row's threshold is ranked from at most, in shared memory
    MOST_SAMPLES: int = 8192
    # Threads of a row's CTA - few enough that several rows share an SM
    LISTING_THREADS: int = 512
    LISTING_WARPS: int = LISTING_THREADS // Warp.SIZE
    # Columns a thread reads in a pass - a CTA's width apart, or side by side in an ordered listing
    COLUMNS_PER_THREAD: int = 8
    # Digits of the sampled threshold settled - the rest left clear, so the threshold sits at or under the sample's rank
    THRESHOLD_PASSES: int = 2


class HopperBucketScan:
    """The CuTeDSL bucket scan on Hopper."""

    # Index queries a CTA holds, a warpgroup's threads, buckets a tile scores, and rows an accumulator holds
    BLOCK_Q: int = 128
    BLOCK_N: int = 64
    MMA_ROWS: int = 64
    WAVES: int = 64
    # Tiles of bucket centroids in flight, which is also the tiles of a pass, and CTAs an SM holds
    STAGES: int = 2
    MIN_BLOCKS: int = 4
    # Words of hits a thread queues in shared memory a pass, one for each of a pass's tiles and row halves
    QUEUE_SLOTS: int = 2 * STAGES
    # Columns of a thread's accumulator fragment, the bits of a queued word a row
    FRAGMENT_COLUMNS: int = BLOCK_N // 4


class BlackwellBucketScan:
    """The CuTeDSL bucket scan on Blackwell."""

    # Index queries a CTA holds - a tensor-memory lane each - and buckets a tile scores
    # BLOCK_N is tuned on B300
    BLOCK_Q: int = 128
    BLOCK_N: int = 128
    # Scores a thread reads from tensor memory at a time and tests by their maximum
    CHUNK: int = 32
    # WAVES is tuned on B300 (Hopper's is 64)
    WAVES: int = 16
    # The warps - groups of four that test a tile in turn, one that scores, one that copies
    # TEST_GROUPS and STAGES are tuned on B300
    TEST_GROUPS: int = 2
    TEST_WARPS: int = 4 * TEST_GROUPS
    MMA_WARP: int = TEST_WARPS
    COPY_WARP: int = TEST_WARPS + 1
    THREADS: int = Warp.SIZE * (TEST_WARPS + 2)
    TESTERS: int = Warp.SIZE * TEST_WARPS
    # Tiles of bucket centroids in flight, accumulators in tensor memory - two a group, so a group tests one while the
    # other is scored - and CTAs an SM holds
    # ACCUMULATORS * BLOCK_N = 512 columns, all of an SM's tensor memory
    # => one CTA an SM (MIN_BLOCKS = 1), which frees its columns for the next CTA at the FREE_BARRIER
    STAGES: int = 4
    ACCUMULATORS: int = 2 * TEST_GROUPS
    MIN_BLOCKS: int = 1
    # Named barriers - 0 is the CTA's own
    # The named barrier the tensor memory's address is published on
    TMEM_BARRIER: int = 1
    FREE_BARRIER: int = 2


class HopperTileSize:
    """The tile the Hopper kernels share."""

    # Buckets a tile holds, and a warpgroup's threads
    TILE: int = 64
    THREADS: int = 128
    WARPS: int = THREADS // Warp.SIZE


class HopperClustering:
    """The CuTeDSL node-bucket clustering on Hopper."""

    # Open buckets a leader pairs alone from its matrix of costs, a row padded a word against bank conflicts
    TAIL: int = HopperTileSize.TILE
    TAIL_STRIDE: int = TAIL + 1
    # The tail's two warps, whose open counts the last step adds
    TAIL_WARPS: int = TAIL // Warp.SIZE
    # Rounds between two checks, as the Triton pairing looks
    CHECK: int = 8
    # Blocks a launch takes at most
    MAX_BLOCKS: int = 256
    # The workspace's int32 parts, each G words but the last - a block's count of open buckets, whether to stop, the counts
    # at the two alternating checks (two parts), the arrival counter of each block's CTAs, and one counter for the leaders
    COUNTS: int = 0
    STOPS: int = 1
    CHECKS: int = 2
    ARRIVALS: int = 4
    LEADERS: int = 5


class HopperWard:
    """The CuTeDSL ward clustering on Hopper."""

    # Pools a block may hold - a 512-key window's keys unpooled, whose sums fill most of a CTA's shared memory
    MOST_POOLS: int = 512
    # At 512 pools the opt-in budget is 227 KB = 232,448 B, and a CTA holds
    #   tiles   rows and columns, [64, 64] fp32 in two splits each                  2 * 32,768 =  65,536 B
    #   sums    512 * (64 + 1 padding) fp32                                                   = 133,120 B
    #   state   square, alignment, size, alive, parent, 5 * 512 * 4                           =  10,240 B
    #   round   best 2 * 512 * 4, cost keys 2 * 512 * 8, mutual 512 * 8                       =  16,384 B
    #   misc    warp totals, tile barrier, tensor memory address                              =      28 B
    # => 225,308 B, 7,140 B under the budget - a four-word padding (8,192 B) would leave 996 B
    # CTAs a cluster may hold - the portable limit
    MOST_CLUSTER_CTAS: int = 8
    # Words a row of sums is padded by, so a warp reading one dim of 16 pools meets 16 banks, not one
    SUMS_PADDING: int = 1
    # Dims a run of the squared length adds before the runs are halved
    RUN: int = 4
