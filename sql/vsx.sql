SET max_bytes_before_external_group_by = 2000000000;

SELECT
    vsx_oid,
    vsx_type,
    arr.1 AS ztf_oid,
    arr.2 AS filter,
    arr.3 AS mjd,
    arr.4 AS mag,
    arr.5 AS magerr
FROM
(
    SELECT
        vsx_oid,
        any(type) AS vsx_type,
        arraySort((arr) -> arr.3, groupArray((ztf_oid, filter, mjd, mag, magerr))) AS arr
    FROM
    (
        SELECT
            oid AS ztf_oid,
            h3index10,
            filter,
            mjd,
            mag,
            magerr,
            vsx_oid,
            type
        FROM ztf.dr3
        INNER JOIN
        (
            SELECT
                oid AS vsx_oid,
                type,
                arrayJoin(h3kRing(h3index10, toUInt8(ceil((1. / 3600.) / h3EdgeAngle(10))))) AS h3index10,
                radeg AS ra_vsx,
                dedeg AS dec_vsx
            FROM vsx.vsx
            WHERE isNotNull(type) AND (v = 0) AND (NOT (isNull(f_min) AND (min < 12.0))) AND (max > 10.0)
        ) AS vsx_wide USING (h3index10)
        WHERE (catflags = 0) AND (greatCircleAngle(ra, dec, ra_vsx, dec_vsx) <= (1. / 3600.))
    )
    GROUP BY vsx_oid
)
INTO OUTFILE '/tmp/ztf-vsx.csv'
FORMAT CSV
